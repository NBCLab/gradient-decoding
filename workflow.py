"""Workflow for running the grdient-decoding analyses"""
import argparse
import itertools
import os
import os.path as op
import pickle
from glob import glob

import mapalign
import nibabel as nib
import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
from gradec.decode import GCLDADecoder, LDADecoder, TermDecoder
from gradec.fetcher import _fetch_features
from gradec.utils import _conform_features

import utils
from performance import (
    _combine_counts,
    _get_ic,
    _get_semantic_similarity,
    _get_tfidf,
    _get_twfrequencies,
    classifier,
)
from segmentation import (
    KDESegmentation,
    KMeansSegmentation,
    PCTLSegmentation,
    compare_segmentations,
    gradient_to_maps,
)

DEC_MODELS = {
    "term": TermDecoder,
    "lda": LDADecoder,
    "gclda": GCLDADecoder,
}


def _get_parser():
    parser = argparse.ArgumentParser(description="Run gradient-decoding workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=4,
        required=False,
        help="CPUs",
    )
    return parser


def hcp_gradient(data_dir, template_dir, principal_gradient_fn, pypackage="mapalign"):
    """1. Functional Connectivity Gradient: Perform gradient decomposition of the group-average
    dense connectome from HCP resting-state fMRI data.

    1.1. HCP S1200 group-averge dense connectome.
    1.2. 64,984 X 64,984 functional connectivity matrices.
    1.3. Compute affinity matrix.
    1.4. Apply diffusion map embedding

    Parameters
    ----------
    none : :obj:``
    Returns
    -------
    None : :obj:``
    """
    output_dir = op.dirname(principal_gradient_fn)
    os.makedirs(output_dir, exist_ok=True)

    gradients_fn = op.join(output_dir, "gradients.npy")
    lambdas_fn = op.join(output_dir, "lambdas.npy")
    if not (op.isfile(gradients_fn) and op.isfile(lambdas_fn)):
        print("\t\tLoading connenctivity mtx and apply Fisher's z-to-r transform...", flush=True)
        dcon_img = nib.load(
            op.join(
                data_dir, "hcp", "HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii"
            )
        )
        dcon_mtx = np.tanh(dcon_img.get_fdata())  # 91,282 X 91,282 grayordinates

        del dcon_img

        print("\t\tApplying diffusion map embedding...", flush=True)
        if pypackage == "mapalign":
            # Calculate affinity matrix
            dcon_mtx = utils.affinity(dcon_mtx, 90)
            gradients, statistics = mapalign.embed.compute_diffusion_map(
                dcon_mtx, alpha=0.5, return_result=True, overwrite=True
            )
            pickle.dump(statistics, open(op.join(output_dir, "statistics.p"), "wb"))
            lambdas = statistics["lambdas"]
        elif pypackage == "brainspace":
            gm = GradientMaps(n_components=10, random_state=0, kernel="cosine", approach="dm")
            gm.fit(dcon_mtx, sparsity=0.9, n_iter=10)
            gradients, lambdas = gm.gradients_, gm.lambdas_

        del dcon_mtx

        np.save(gradients_fn, gradients)
        np.save(lambdas_fn, lambdas)
    else:
        print("\t\tLoading diffusion map embedding...", flush=True)
        gradients = np.load(gradients_fn)
        lambdas = np.load(lambdas_fn)

    utils.plot_dm_results(lambdas, output_dir)

    print("\t\tExporting gradient to NIFTI and GIFTI files...", flush=True)
    # Load subcortical volume
    subcortical_fn = op.join(template_dir, "rois-subcortical_mni152_mask.nii.gz")
    subcort_img = nib.load(subcortical_fn)

    utils._gradient_to_nifti(gradients, subcort_img, output_dir)
    return utils._gradient_to_gifti(
        gradients,
        subcort_img,
        principal_gradient_fn,
        output_dir,
    )


def gradient_segmentation(gradient, grad_seg_fn, n_segments):
    """2. Segmentation and Gradient Maps: Evaluate three different segmentation approaches to
    split the gradient spectrum into a finite number of brain maps.

    2.1. Segment the gradient into k ≥ 3 segments using:
        - Percentile Segmentation
        - K-Means
        - KDE
    2.2. Transform KDE segmented gradient maps to activation maps.


    Parameters
    ----------
    none : :obj:``
    Returns
    -------
    None : :obj:``
    """
    print("\t2.1. Segment the gradient into k ≥ 3 segments.", flush=True)
    grad_seg_dict = {}
    output_dir = op.dirname(grad_seg_fn)

    for method in ["Percentile", "KMeans", "KDE"]:
        spc_output_dir = op.join(output_dir, method.lower())
        results_fn = op.join(spc_output_dir, f"{method.lower()}_results.pkl")
        if not op.isfile(results_fn):
            if method == "Percentile":
                # Percentile Segmentation
                print("\t\tRunning Percentile Segmentation...", flush=True)
                segment_method = PCTLSegmentation(results_fn, n_segments)
            elif method == "KMeans":
                # K-Means
                print("\t\tRunning K-Means Segmentation...", flush=True)
                segment_method = KMeansSegmentation(results_fn, n_segments)
            elif method == "KDE":
                # KDE
                print("\t\tRunning KDE Segmentation...", flush=True)
                segment_method = KDESegmentation(results_fn, n_segments)

            results_dict = segment_method.fit(gradient)
        else:
            print(f"\t\tLoading Results from {method} Segmentation...", flush=True)
            with open(results_fn, "rb") as results_file:
                results_dict = pickle.load(results_file)

        segments, labels, peaks = (
            results_dict["segments"],
            results_dict["labels"],
            results_dict["peaks"],
        )
        assert len(labels) == n_segments

        # Save labels to calculate silhouette measures
        if method == "Percentile":
            percent_labels = labels.copy()
        elif method == "KMeans":
            kmeans_labels = labels.copy()
        elif method == "KDE":
            kde_labels = labels.copy()

        # 2.2. Transform KDE segmented gradient maps to activation maps.
        print(f"\t\tTransforming {method} segmented grad maps to activation maps...", flush=True)
        grad_seg_dict = gradient_to_maps(method, segments, peaks, grad_seg_dict, spc_output_dir)

    with open(grad_seg_fn, "wb") as grad_segments_file:
        pickle.dump(grad_seg_dict, grad_segments_file)

    # Silhouette measures
    silhouette_df_fn = op.join(output_dir, "silhouette_scores.csv")
    # if not op.isfile(silhouette_df_fn):
    print("\tCalculating Silhouette measures...", flush=True)
    compare_segmentations(gradient, percent_labels, kmeans_labels, kde_labels, silhouette_df_fn)

    return grad_seg_dict


def gradient_decoding(data_dir, grad_seg_dict, output_dir, n_cores):
    """3. Meta-Analytic Functional Decoding: Implement six different decoding strategies and
    perform an optimization test to identify the segment size to split the gradient for each
    strategy.

    3.1. Generate meta-analytic maps.
    3.2. Calculate correlation between meta-analytic maps and unthresholded gradient activation
         maps for each strategy.
    3.3. Select a set with optimal segment size for each strategy.

    Parameters
    ----------
    none : :obj:``
    Returns
    -------
    None : :obj:``
    """
    N_SAMPLES = 1000
    dset_nms = ["neurosynth", "neuroquery"]
    model_nms = ["term", "lda", "gclda"]
    segnt_nms = ["Percentile", "KMeans", "KDE"]
    for dset_nm, model_nm, segnt_nm in itertools.product(dset_nms, model_nms, segnt_nms):
        corr_dir = op.join(output_dir, f"{dset_nm}_{model_nm}_corr_{segnt_nm}")
        os.makedirs(corr_dir, exist_ok=True)

        grad_segments = grad_seg_dict[f"{segnt_nm.lower()}_grad_segments"]
        for grad_maps in grad_segments:
            decode = DEC_MODELS[model_nm](n_samples=N_SAMPLES, data_dir=data_dir, n_cores=n_cores)
            decode.fit(dset_nm)
            corrs_df, pvals_df, pvals_FDR_df = decode.transform(grad_maps, method="correlation")

            n_segments = len(grad_maps)
            corrs_df.to_csv(op.join(corr_dir, f"corrs_{n_segments:02d}.csv"))
            pvals_df.to_csv(op.join(corr_dir, f"pvals_{n_segments:02d}.csv"))
            pvals_FDR_df.to_csv(op.join(corr_dir, f"pvals-FDR_{n_segments:02d}.csv"))


def decoding_performance(data_dir, dec_data_dir, output_dir):
    """4. Performance of Decoding Strategies: Evaluate the different decoding strategies using
    multiple metrics to compare relative performance.

    4.1. Compare correlation profiles.
    4.2. Compare semantic similarity metrics.
        - Information Content (IC)
        - TF-IDF
    4.3. Compare SNR.

    Parameters
    ----------
    none : :obj:``
    Returns
    -------
    None : :obj:``
    """
    os.makedirs(output_dir, exist_ok=True)
    models_dir = op.join(data_dir, "models")

    counts_df_fn = op.join(output_dir, "nsnq_counts.tsv")
    if not op.isfile(counts_df_fn):
        # Generate counts of combined dataset
        counts_df = _combine_counts(output_dir)
        counts_df.to_csv(counts_df_fn, sep="\t")
    else:
        counts_df = pd.read_csv(counts_df_fn, delimiter="\t", index_col="id")

    ic_df_fn = op.join(output_dir, "nsnq_ic.tsv")
    if not op.isfile(ic_df_fn):
        ic_df = _get_ic(counts_df)
        ic_df.to_csv(ic_df_fn, sep="\t", index=False)
    else:
        ic_df = pd.read_csv(ic_df_fn, delimiter="\t")

    tfidf_df_fn = op.join(output_dir, "nsnq_tfidf.tsv")
    if not op.isfile(tfidf_df_fn):
        tfidf_df = _get_tfidf(counts_df)
        tfidf_df.to_csv(tfidf_df_fn, sep="\t")
    else:
        tfidf_df = pd.read_csv(tfidf_df_fn, delimiter="\t", index_col="id")

    frequency_threshold = 0.001
    N_TOP_WORDS = 3
    dset_nms = ["neurosynth", "neuroquery"]
    model_nms = ["term", "lda", "gclda"]
    segnt_nms = ["Percentile", "KMeans", "KDE"]
    (
        max_corr_lst,
        idx_lst,
        feature_lst,
        max_pval_lst,
        max_fdr_pval_lst,
        segments_lst,
        method_lst,
        seg_sol_lst,
        ic_lst,
        tfidf_lst,
        classification_lst,
        mean_corr_lst,
        mean_seg_sol_lst,
        mean_ic_lst,
        mean_tfidf_lst,
        mean_method_lst,
        snr_lst,
    ) = ([] for _ in range(17))
    for dset_nm, model_nm in itertools.product(dset_nms, model_nms):
        # Get topic-wise frequencies
        # We don't need weights for classifying terms
        frequencies = (
            _get_twfrequencies(dset_nm, model_nm, N_TOP_WORDS, models_dir)
            if model_nm in ["lda", "gclda"]
            else None
        )

        # Get and classify features
        features = _fetch_features(dset_nm, model_nm, data_dir=data_dir)
        features = _conform_features(features, N_TOP_WORDS, model_nm)
        features_arr = np.array(features)
        features_classified = classifier(
            features_arr, N_TOP_WORDS, frequencies, dset_nm, model_nm, data_dir
        )

        for segnt_nm in segnt_nms:
            corr_dir = op.join(dec_data_dir, f"{dset_nm}_{model_nm}_corr_{segnt_nm}")
            corr_lst = sorted(glob(op.join(corr_dir, "corrs_*.csv")))
            pval_lst = sorted(glob(op.join(corr_dir, "pvals_*.csv")))
            pval_fdr_lst = sorted(glob(op.join(corr_dir, "pvals-FDR_*.csv")))

            for corr_fn, pval_fn, pval_fdr_fn in zip(corr_lst, pval_lst, pval_fdr_lst):
                corr_df = pd.read_csv(corr_fn, index_col="feature")
                pval_df = pd.read_csv(pval_fn, index_col="feature")
                pval_fdr_df = pd.read_csv(pval_fdr_fn, index_col="feature")

                # Get maximum correlation and corresponding feature
                max_df = corr_df.idxmax()
                max_idx = corr_df.index.get_indexer(max_df.values)
                max_corr = np.diag(corr_df.loc[max_df.values, max_df.index])
                max_pval = np.diag(pval_df.loc[max_df.values, max_df.index])
                max_fdr_pval = np.diag(pval_fdr_df.loc[max_df.values, max_df.index])
                max_features = max_df.values
                max_feature_clss = features_classified[max_idx]

                # Get information content, and tfidf per max features
                n_seg = corr_df.shape[1]
                segments = np.arange(1, n_seg + 1)
                temp_ic_lst, temp_tfidf_lst = _get_semantic_similarity(
                    model_nm,
                    ic_df,
                    tfidf_df,
                    max_features,
                    frequency_threshold,
                    N_TOP_WORDS,
                )

                # Calculate SNR per max features
                snr = sum(np.array(max_feature_clss) == "Functional") / len(max_feature_clss)

                # Append values for performance DF
                method_lst.append([f"{model_nm}_{dset_nm}_{segnt_nm}"] * n_seg)
                segments_lst.append(segments)
                seg_sol_lst.append([f"{n_seg}"] * n_seg)
                max_corr_lst.append(max_corr)
                max_pval_lst.append(max_pval)
                max_fdr_pval_lst.append(max_fdr_pval)
                idx_lst.append(max_idx)
                feature_lst.append(max_features)
                ic_lst.append(temp_ic_lst)
                tfidf_lst.append(temp_tfidf_lst)
                classification_lst.append(max_feature_clss)

                # Append values for average performance DF
                mean_method_lst.append(f"{model_nm}_{dset_nm}_{segnt_nm}")
                mean_seg_sol_lst.append(f"{n_seg}")
                mean_corr_lst.append(np.mean(max_corr))
                mean_ic_lst.append(np.mean(temp_ic_lst))
                mean_tfidf_lst.append(np.mean(temp_tfidf_lst))
                snr_lst.append(snr)

    # Initialize performance DF
    data_df = pd.DataFrame()
    data_df["method"] = np.hstack(method_lst)
    data_df["segment"] = np.hstack(segments_lst)
    data_df["segment_solution"] = np.hstack(seg_sol_lst)
    data_df["max_corr"] = np.hstack(max_corr_lst)
    data_df["pvalue"] = np.hstack(max_pval_lst)
    data_df["fdr_pvalue"] = np.hstack(max_fdr_pval_lst)
    data_df["corr_idx"] = np.hstack(idx_lst)
    data_df["features"] = np.hstack(feature_lst)
    data_df["information_content"] = np.hstack(ic_lst)
    data_df["tfidf"] = np.hstack(tfidf_lst)
    data_df["classification"] = np.hstack(classification_lst)

    mean_data_df = pd.DataFrame()
    mean_data_df["method"] = np.hstack(mean_method_lst)
    mean_data_df["segment_solution"] = mean_seg_sol_lst
    mean_data_df["max_corr"] = np.hstack(mean_corr_lst)
    mean_data_df["ic"] = np.hstack(mean_ic_lst)
    mean_data_df["tfidf"] = np.hstack(mean_tfidf_lst)
    mean_data_df["snr"] = np.hstack(snr_lst)

    return data_df, mean_data_df


def decoding_results():
    """5. Visualization of the Decoded Maps: Investigate four visualization approaches for
    reporting decoded gradient results and assess each via a community survey.

    5.1.

    Parameters
    ----------
    none : :obj:``
    Returns
    -------
    None : :obj:``
    """

    return None


def main(project_dir, n_cores):
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)

    # Define Paths
    # =============
    results_dir = op.join(project_dir, "results")
    data_dir = op.join(project_dir, "data")
    gradient_dir = op.join(results_dir, "gradient")
    segmentation_dir = op.join(results_dir, "segmentation")
    decoding_dir = op.join(results_dir, "decoding")
    performance_dir = op.join(results_dir, "performance")
    templates_dir = op.join(project_dir, "data", "templates")
    # figure_dir = op.join(project_dir, "results", "decoding_results")

    N_SEGMENTS = 30
    N_DSETS = 2
    N_MODELS = 3
    N_SEGMODELS = 3

    # Run Workflow
    # =============
    # 1. Functional Connectivity Gradient
    print("1. Functional Connectivity Gradient", flush=True)
    principal_gradient_fn = op.join(gradient_dir, "principal_gradient.npy")
    if not op.isfile(principal_gradient_fn):
        principal_gradient = hcp_gradient(data_dir, templates_dir, principal_gradient_fn)
    else:
        print("\tGradient file exists. Loading principal gradient...", flush=True)
        principal_gradient = np.load(principal_gradient_fn)

    # 2. Segmentation and Gradient Maps
    print("2. Segmentation and Gradient Maps", flush=True)
    grad_seg_fn = op.join(segmentation_dir, "grad_segments.pkl")
    if not op.isfile(grad_seg_fn):
        grad_seg_dict = gradient_segmentation(principal_gradient, grad_seg_fn, N_SEGMENTS)
    else:
        print("\tGradient dict exists. Loading segmented gradient...", flush=True)
        grad_segments_file = open(grad_seg_fn, "rb")
        grad_seg_dict = pickle.load(grad_segments_file)

    # 3. Meta-Analytic Functional Decoding
    print("3. Meta-Analytic Functional Decoding", flush=True)
    n_result_files = len(glob(op.join(decoding_dir, "*", "*.csv")))
    if n_result_files < N_DSETS * N_MODELS * N_SEGMODELS * N_SEGMENTS * 3:
        # if n_result_files < 1443:
        gradient_decoding(
            data_dir,
            grad_seg_dict,
            results_dir,
            n_cores,
        )
    else:
        print("\tDecoding CSV exist. Skipping functional decoding...", flush=True)

    # 4. Performance of Decoding Strategies
    print("4. Performance of Decoding Strategies", flush=True)
    performance_fn = op.join(performance_dir, "performance.tsv")
    performance_average_fn = op.join(performance_dir, "performance_average.tsv")
    if not op.isfile(performance_fn) or not op.isfile(performance_average_fn):
        performance_df, performance_average_df = decoding_performance(
            data_dir, decoding_dir, performance_dir
        )

        performance_df.to_csv(performance_fn, sep="\t")
        performance_average_df.to_csv(performance_average_fn, sep="\t")

    # 5. Visualization of the Decoded Maps
    # print("5. Visualization of the Decoded Maps", flush=True)
    # decoding_results(performance_df)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
