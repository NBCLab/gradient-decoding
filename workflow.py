"""Workflow for running the grdient-decoding analyses"""
import argparse
import gzip
import os
import os.path as op
import pickle
from glob import glob

import mapalign
import nibabel as nib
import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
from neuromaps import transforms
from nibabel import GiftiImage
from nibabel.gifti import GiftiDataArray
from nilearn import masking
from nimare.annotate.gclda import GCLDAModel
from nimare.dataset import Dataset
from nimare.decode.continuous import CorrelationDecoder
from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma import mkda
from nimare.stats import pearson
from surfplot.utils import add_fslr_medial_wall

import utils
from decoding import _get_counts, annotate_lda, gen_nullsamples
from performance import classifier
from segmentation import (
    KDESegmentation,
    KMeansSegmentation,
    PCTLSegmentation,
    compare_segmentations,
    gradient_to_maps,
)


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
    full_vertices = 64984
    hemi_vertices = int(full_vertices / 2)
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
    subcort_dat = subcort_img.get_fdata()
    subcort_mask = subcort_dat != 0
    n_subcort_vox = np.where(subcort_mask)[0].shape[0]

    n_gradients = gradients.shape[1]
    for i in range(n_gradients):
        # Exclude 31,870 voxels form subcortical structures as represented in volumetric space
        # = 59,412 excluding the medial wall
        subcort_grads = gradients[gradients.shape[0] - n_subcort_vox :, i]
        cort_grads = gradients[: gradients.shape[0] - n_subcort_vox, i]

        if i == 0:
            # Save principal gradient
            principal_gradient = cort_grads.copy()
            np.save(principal_gradient_fn, principal_gradient)

        # Add the medial wall: 32,492 X 32,492 grayordinates = 64,984, for visualization purposes
        # Get left and rigth hemisphere gradient scores, and insert 0's where medial wall is
        grad_map_full = add_fslr_medial_wall(cort_grads, split=False)
        gradients_lh, gradients_rh = grad_map_full[:hemi_vertices], grad_map_full[hemi_vertices:]

        grad_img_lh = GiftiImage()
        grad_img_rh = GiftiImage()
        grad_img_lh.add_gifti_data_array(GiftiDataArray(gradients_lh))
        grad_img_rh.add_gifti_data_array(GiftiDataArray(gradients_rh))

        subcort_grads_fn = op.join(
            output_dir,
            "source-jperaza2022_desc-fcG{:02d}_space-MNI152_den-2mm_feature.nii.gz".format(i),
        )
        gradients_lh_fn = op.join(
            output_dir,
            "source-jperaza2022_desc-fcG{:02d}_space-fsLR_den-32k_hemi-L_feature.func.gii".format(
                i
            ),
        )
        gradients_rh_fn = op.join(
            output_dir,
            "source-jperaza2022_desc-fcG{:02d}_space-fsLR_den-32k_hemi-R_feature.func.gii".format(
                i
            ),
        )

        # Write subcortical gradient to Nifti file
        new_subcort_dat = np.zeros_like(subcort_dat)
        new_subcort_dat[subcort_mask] = subcort_grads
        new_subcort_img = nib.Nifti1Image(new_subcort_dat, subcort_img.affine, subcort_img.header)
        new_subcort_img.to_filename(subcort_grads_fn)

        # Write cortical gradient to Gifti file
        nib.save(grad_img_lh, gradients_lh_fn)
        nib.save(grad_img_rh, gradients_rh_fn)

    return principal_gradient


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
            results_file = open(results_fn, "rb")
            results_dict = pickle.load(results_file)
            results_file.close()

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

    # Save results
    grad_segments_file = open(grad_seg_fn, "wb")
    pickle.dump(grad_seg_dict, grad_segments_file)
    grad_segments_file.close()

    # Silhouette measures
    silhouette_df_fn = op.join(output_dir, "silhouette_scores.csv")
    # if not op.isfile(silhouette_df_fn):
    print("\tCalculating Silhouette measures...", flush=True)
    compare_segmentations(gradient, percent_labels, kmeans_labels, kde_labels, silhouette_df_fn)

    return grad_seg_dict


def gradient_decoding(data_dir, output_dir, grad_seg_dict, n_cores):
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
    neuromaps_dir = op.join(data_dir, "neuromaps-data")
    ma_data_dir = op.join(data_dir, "meta-analysis")
    os.makedirs(ma_data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    n_vertices = 59412  # TODO: 59412 is harcoded here
    n_permutations = 1000
    n_topics = 200
    nullsamples_fn = op.join(output_dir, "null_samples_fslr.npy")
    if not op.isfile(nullsamples_fn):
        nullsamples = gen_nullsamples(neuromaps_dir, n_samples=n_permutations, n_cores=n_cores)
        np.save(nullsamples_fn, nullsamples)
    else:
        nullsamples = np.load(nullsamples_fn)

    # dset_names = ["neurosynth", "neuroquery"]
    # sources = ["term", "lda", "gclda"]
    # methods = ["Percentile", "KMeans", "KDE"]
    dset_names = ["neuroquery"]
    sources = ["lda"]
    methods = ["Percentile"]
    for dset_name in dset_names:
        dset_fn = os.path.join(ma_data_dir, f"{dset_name}_dataset.pkl.gz")
        if not os.path.isfile(dset_fn):
            print(f"\tFetching {dset_name} database...", flush=True)
            if dset_name == "neurosynth":
                files = fetch_neurosynth(
                    data_dir=ma_data_dir,
                    version="7",
                    overwrite=False,
                    source="abstract",
                    vocab="terms",
                )
            elif dset_name == "neuroquery":
                files = fetch_neuroquery(
                    data_dir=data_dir,
                    version="1",
                    overwrite=False,
                    source="combined",
                    vocab="neuroquery6308",
                    type="tfidf",
                )

            dataset_db = files[0]

            dset = convert_neurosynth_to_dataset(
                coordinates_file=dataset_db["coordinates"],
                metadata_file=dataset_db["metadata"],
                annotations_files=dataset_db["features"],
            )
            dset.save(dset_fn)
        else:
            print(f"\tLoading {dset_name} database...", flush=True)
            dset = Dataset.load(dset_fn)

        # Check whether dset contain text for LDA-based models
        if dset_name == "neurosynth":
            if "abstract" in dset.texts:
                # Download Neurosynth abstract
                print(f"\t\tDataset {dset_name} contains abstract", flush=True)
            else:
                print(f"\t\tDownloading abstract to {dset_name} dset", flush=True)
                dset = download_abstracts(dset, "jpera054@fiu.edu")
                dset.save(dset_fn)
        elif dset_name == "neuroquery":
            # LDA model will be run on word_counts, so the text is not needed
            pass

        if dset.basepath is None:
            basepath_path = op.join(ma_data_dir, f"{dset_name}_basepath")
            os.makedirs(basepath_path, exist_ok=True)
            dset.update_path(basepath_path)

        for source in sources:
            if source == "term":
                # Term-based meta-analysis
                if dset_name == "neurosynth":
                    feature_group = "terms_abstract_tfidf"
                elif dset_name == "neuroquery":
                    feature_group = "neuroquery6308_combined_tfidf"
                frequency_threshold = 0.001

            if source == "lda":
                # LDA-based meta-analysis
                feature_group = f"LDA{n_topics}"
                frequency_threshold = 0.05

            if (source == "term") or (source == "lda"):
                decoder_fn = op.join(output_dir, f"{source}_{dset_name}_decoder.pkl.gz")
                if not op.isfile(decoder_fn):
                    if source == "lda":
                        print(f"\tRunning LDA model on {dset_name}...", flush=True)
                        # n_cores=1 for LDA.
                        # See: https://github.com/scikit-learn/scikit-learn/issues/8943
                        new_dset_fn = dset_fn.split("_dataset.pkl.gz")[0] + "_lda_dataset.pkl.gz"
                        if not op.isfile(new_dset_fn):
                            lda_based_model_fn = op.join(
                                output_dir, f"lda_{dset_name}_model.pkl.gz"
                            )
                            dset = annotate_lda(
                                dset,
                                dset_name,
                                ma_data_dir,
                                lda_based_model_fn,
                                n_topics=n_topics,
                                n_cores=1,
                            )
                            dset.save(new_dset_fn)
                        else:
                            dset = Dataset.load(new_dset_fn)

                    print(
                        f"\tPerforming {source.upper()}-based meta-analysis on {dset_name}...",
                        flush=True,
                    )
                    decoder = CorrelationDecoder(
                        frequency_threshold=frequency_threshold,
                        meta_estimator=mkda.MKDAChi2,
                        feature_group=feature_group,
                        target_image="z_desc-specificity",
                        n_cores=n_cores,
                    )
                    decoder.fit(dset)
                    decoder.save(decoder_fn, compress=True)
                else:
                    print(
                        f"\tLoading {source.upper()}-based meta-analytic maps from {dset_name}...",
                        flush=True,
                    )
                    decoder_file = gzip.open(decoder_fn, "rb")
                    decoder = pickle.load(decoder_file)

            elif source == "gclda":
                # GCLDA-based meta-analysis
                gclda_based_model_fn = op.join(output_dir, f"gclda_{dset_name}_model.pkl.gz")
                if not op.isfile(gclda_based_model_fn):
                    print(f"\tRunning GCLDA model on {dset_name}...", flush=True)
                    n_iters = 20000
                    counts_df = _get_counts(dset, dset_name, ma_data_dir)
                    counts_df_fn = op.join(output_dir, f"{dset_name}_counts.tsv")
                    counts_df.to_csv(counts_df_fn, sep="\t")

                    gclda_model = GCLDAModel(
                        counts_df,
                        dset.coordinates,
                        mask=dset.masker.mask_img,
                        n_topics=n_topics,
                        n_regions=4,
                        symmetric=True,
                        n_cores=n_cores,
                    )
                    gclda_model.fit(n_iters=n_iters, loglikely_freq=100)
                    gclda_model.save(gclda_based_model_fn, compress=True)
                else:
                    print(
                        f"\tLoading GCLDA-based meta-analytic maps from {dset_name}...", flush=True
                    )
                    gclda_decoder_file = gzip.open(gclda_based_model_fn, "rb")
                    gclda_model = pickle.load(gclda_decoder_file)

            # Get meta-analytic maps
            if (source == "term") or (source == "lda"):
                meta_arr = decoder.images_
            elif source == "gclda":
                meta_arr = gclda_model.p_voxel_g_topic_.T

            n_metamaps = meta_arr.shape[0]
            meta_map_fn = op.join(output_dir, f"{source}_{dset_name}_metamaps.npy")
            meta_null_fn = op.join(output_dir, f"{source}_{dset_name}_metamaps-nulls.npy")
            if op.isfile(meta_map_fn) and op.isfile(meta_null_fn):
                print("\tLoading meta-analytic and null maps...", flush=True)
                meta_maps_fslr_arr = np.load(meta_map_fn)
                if n_metamaps > 200:
                    meta_maps_permuted_arr = np.memmap(
                        meta_null_fn,
                        dtype="float32",
                        mode="r",
                        shape=(n_metamaps, n_vertices, n_permutations),
                    )
                else:
                    meta_maps_permuted_arr = np.load(meta_null_fn)
            else:
                print("\tTransforming meta-analytic and null maps to fsLR...", flush=True)
                meta_maps_fslr_arr = np.zeros((n_metamaps, n_vertices))
                if n_metamaps > 200:
                    meta_maps_permuted_arr = np.memmap(
                        meta_null_fn,
                        dtype="float32",
                        mode="w+",
                        shape=(n_metamaps, n_vertices, n_permutations),
                    )
                else:
                    meta_maps_permuted_arr = np.zeros((n_metamaps, n_vertices, n_permutations))

                for metamap_i in range(n_metamaps):
                    print(metamap_i, flush=True)
                    fslr_dir = op.join(output_dir, f"{source}_{dset_name}_fslr")
                    os.makedirs(fslr_dir, exist_ok=True)

                    if (source == "gclda") or (source == "lda"):
                        desc_name = "desc-topic{:03d}".format(metamap_i + 1)
                    elif source == "term":
                        desc_name = "desc-term{:04d}".format(metamap_i + 1)

                    meta_map_lh_fn = op.join(
                        fslr_dir,
                        f"source-{source}_dset-{dset_name}_{desc_name}"
                        "_space-fsLR_den-32k_hemi-L_feature.func.gii",
                    )
                    meta_map_rh_fn = op.join(
                        fslr_dir,
                        f"source-{source}_dset-{dset_name}_{desc_name}"
                        "_space-fsLR_den-32k_hemi-R_feature.func.gii",
                    )

                    if op.isfile(meta_map_lh_fn) and op.isfile(meta_map_rh_fn):
                        meta_map_lh = nib.load(meta_map_lh_fn)
                        meta_map_rh = nib.load(meta_map_rh_fn)
                    else:
                        if (source == "term") or (source == "lda"):
                            meta_map = decoder.masker.inverse_transform(meta_arr[metamap_i, :])
                        elif source == "gclda":
                            meta_map = masking.unmask(meta_arr[metamap_i, :], gclda_model.mask)

                        meta_map_lh, meta_map_rh = transforms.mni152_to_fslr(meta_map)

                        meta_map_lh, meta_map_rh = utils.zero_fslr_medial_wall(
                            meta_map_lh, meta_map_rh, neuromaps_dir
                        )

                        # Write cortical gradient to Gifti files
                        nib.save(meta_map_lh, meta_map_lh_fn)
                        nib.save(meta_map_rh, meta_map_rh_fn)

                        del meta_map
                    """
                    meta_map_arr_lh = meta_map_lh.agg_data()
                    meta_map_arr_rh = meta_map_rh.agg_data()

                    meta_map_fslr = utils.rm_fslr_medial_wall(
                        meta_map_arr_lh, meta_map_arr_rh, neuromaps_dir
                    )

                    meta_maps_fslr_arr[metamap_i, :] = meta_map_fslr
                    meta_maps_permuted_arr[metamap_i, :, :] = meta_map_fslr[nullsamples]
                    """

                np.save(meta_map_fn, meta_maps_fslr_arr)
                if n_metamaps > 200:
                    meta_maps_permuted_arr.flush()
                else:
                    np.save(meta_null_fn, meta_maps_permuted_arr)

                del (
                    meta_arr,
                    meta_map_fslr,
                    nullsamples,
                    meta_map_arr_lh,
                    meta_map_arr_rh,
                    meta_map_lh,
                    meta_map_rh,
                )

            if (source == "term") or (source == "lda"):
                del decoder
            elif source == "gclda":
                del gclda_model

            null_dir = op.join(output_dir, f"{source}_{dset_name}_null")
            os.makedirs(null_dir, exist_ok=True)
            for perm_i in range(n_permutations):
                meta_null_i_fn = op.join(
                    null_dir, "{}_{}_metanull-{:04d}.npy".format(source, dset_name, perm_i + 1)
                )
                if not op.isfile(meta_null_i_fn):
                    np.save(meta_null_i_fn, meta_maps_permuted_arr[:, :, perm_i])

            del meta_maps_permuted_arr

            # Correlate meta-analytic maps with segmented maps
            for method in methods:
                corr_dir = op.join(output_dir, f"{source}_{dset_name}_corr_{method}")
                os.makedirs(corr_dir, exist_ok=True)

                grad_segments = grad_seg_dict[f"{method.lower()}_grad_segments"]

                n_segmentations = len(grad_segments)
                for segmentation_i in range(n_segmentations):
                    n_segments = len(grad_segments[segmentation_i])

                    corrs_sol_fn = op.join(corr_dir, "{:02d}_corr.npy".format(segmentation_i + 3))
                    corrs_null_fn = op.join(corr_dir, "{:02d}_null.npy".format(segmentation_i + 3))
                    corrs_pval_fn = op.join(corr_dir, "{:02d}_pval.npy".format(segmentation_i + 3))

                    if op.isfile(corrs_sol_fn) and op.isfile(corrs_null_fn):
                        corrs_sol_arr = np.load(corrs_sol_fn)
                        corrs_null_arr = np.load(corrs_null_fn)
                    else:
                        corrs_sol_arr = np.zeros((n_segments, n_metamaps))
                        corrs_null_arr = np.zeros((n_segments, n_metamaps, n_permutations))
                        for segment_i in range(n_segments):
                            corrs_sol_arr[segment_i, :] = pearson(
                                grad_segments[segmentation_i][segment_i], meta_maps_fslr_arr
                            )

                            # Claculate null correlation coeficients
                            for perm_i in range(n_permutations):
                                meta_null_i_fn = op.join(
                                    null_dir,
                                    "{}_{}_metanull-{:04d}.npy".format(
                                        source, dset_name, perm_i + 1
                                    ),
                                )
                                meta_null_arr_i = np.load(meta_null_i_fn)

                                corrs_null_arr[segment_i, :, perm_i] = pearson(
                                    grad_segments[segmentation_i][segment_i],
                                    meta_null_arr_i,
                                )

                        np.save(corrs_sol_fn, corrs_sol_arr)
                        np.save(corrs_null_fn, corrs_null_arr)

                    if not op.isfile(corrs_pval_fn):
                        corrs_pval_arr = np.zeros((n_segments, n_metamaps))
                        # Calculate p-value of correlations
                        for segment_i in range(n_segments):
                            for metamap_i in range(n_metamaps):
                                true_corr = corrs_sol_arr[segment_i, metamap_i]
                                null_corr = corrs_null_arr[segment_i, metamap_i, :]

                                if true_corr > 0:
                                    summation = null_corr[null_corr > true_corr].sum()
                                else:
                                    summation = null_corr[null_corr < true_corr].sum()

                                p_value = abs(summation / n_permutations)
                                corrs_pval_arr[segment_i, metamap_i] = p_value

                        np.save(corrs_pval_fn, corrs_pval_arr)

    return None


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
    ma_data_dir = op.join(data_dir, "meta-analysis")
    class_data_dir = op.join(data_dir, "classification")

    methods = ["Percentile", "KMeans", "KDE"]
    dset_names = ["neurosynth", "neuroquery"]
    # models = ["lda", "gclda"]
    # methods = ["Percentile"]
    # dset_names = ["neuroquery"]
    models = ["term"]
    max_corr_lst = []
    idx_lst = []
    feature_lst = []
    max_pval_lst = []
    segments_lst = []
    method_lst = []
    seg_sol_lst = []
    ic_lst = []
    tfidf_lst = []
    classification_lst = []
    for dset_name in dset_names:
        dset_fn = os.path.join(ma_data_dir, f"{dset_name}_dataset.pkl.gz")
        dset = Dataset.load(dset_fn)

        counts_df_fn = op.join(dec_data_dir, f"{dset_name}_counts.tsv")
        ic_df_fn = op.join(dec_data_dir, f"{dset_name}_ic.tsv")
        tfidf_df_fn = op.join(dec_data_dir, f"{dset_name}_tfidf.tsv")

        if not op.isfile(counts_df_fn):
            counts_df = _get_counts(dset, dset_name, ma_data_dir)
            # counts_df = counts_df.sort_values(by="id")  # To match matrix in dset.annotations
            counts_df.to_csv(counts_df_fn, sep="\t")
        else:
            counts_df = pd.read_csv(counts_df_fn, delimiter="\t", index_col="id")

        if not op.isfile(ic_df_fn):
            p_t_d = counts_df.div(counts_df.sum(axis=1), axis=0)
            ic_df = -np.log(p_t_d)
            ic_df = ic_df.replace([np.inf, -np.inf], 0)
            ic_df.to_csv(ic_df_fn, sep="\t")
        else:
            ic_df = pd.read_csv(ic_df_fn, delimiter="\t", index_col="id")

        if not op.isfile(tfidf_df_fn):
            tfidf_df = dset.annotations.set_index("id")
            tfidf_df = tfidf_df[tfidf_df.index.isin(ic_df.index)]
            tfidf_df.to_csv(tfidf_df_fn, sep="\t")
        else:
            tfidf_df = pd.read_csv(tfidf_df_fn, delimiter="\t", index_col="id")

        for model in models:
            if model == "lda":
                decoder_fn = op.join(dec_data_dir, f"lda_{dset_name}_decoder.pkl.gz")
                decoder_file = gzip.open(decoder_fn, "rb")
                decoder = pickle.load(decoder_file)
                feature_names = decoder.features_
                features = [f.split("__")[-1] for f in feature_names]
            elif model == "gclda":
                model_fn = op.join(dec_data_dir, f"{model}_{dset_name}_model.pkl.gz")
                model_file = gzip.open(model_fn, "rb")
                decoder = pickle.load(model_file)
                topic_word_weights = decoder.p_word_g_topic_
                n_topics = topic_word_weights.shape[1]
                vocabulary = np.array(decoder.vocabulary)
                sorted_weights_idxs = np.argsort(-topic_word_weights, axis=0)
                feature_names = [
                    "_".join(vocabulary[sorted_weights_idxs[:, topic_i]][:3])
                    for topic_i in range(n_topics)
                ]
                features = [f"{i + 1}_{feature_names[i]}" for i in range(n_topics)]
            elif model == "term":
                feature_group = (
                    "terms_abstract_tfidf"
                    if dset_name == "neurosynth"
                    else "neuroquery6308_combined_tfidf"
                )
                feature_names = tfidf_df.columns.values
                feature_names = [f for f in feature_names if f.startswith(feature_group)]
                features = [f.split("__")[-1] for f in feature_names]

            feature_names = np.array(feature_names)
            features_arr = np.array(features)
            features_classified = classifier(features_arr, dset, model, dset_name, class_data_dir)

            for method in methods:
                corr_dir = op.join(dec_data_dir, f"{model}_{dset_name}_corr_{method}")
                corr_lst = sorted(glob(op.join(corr_dir, "*_corr.npy")))
                pval_lst = sorted(glob(op.join(corr_dir, "*_pval.npy")))

                # plot_df = pd.DataFrame()
                for file_i, corr_file in enumerate(corr_lst):
                    corr_arr = np.load(corr_file)
                    pval_arr = np.load(pval_lst[file_i])

                    max_idx = corr_arr.argmax(axis=1)

                    max_corr = corr_arr[np.arange(corr_arr.shape[0]), max_idx]
                    max_pval = pval_arr[np.arange(pval_arr.shape[0]), max_idx]
                    max_features = features_arr[max_idx]
                    max_feature_names = feature_names[max_idx]
                    max_feature_clss = features_classified[max_idx]
                    n_seg = max_corr.shape[0]
                    segments = np.arange(1, n_seg + 1)
                    # segments = segments/segments.max()
                    if model == "term":
                        for max_feature, max_feature_name, max_feature_cl in zip(
                            max_features, max_feature_names, max_feature_clss
                        ):
                            include_rows = tfidf_df[max_feature_name] >= 0.001
                            include_ic = ic_df[max_feature][include_rows]
                            include_tfidf = tfidf_df[max_feature_name][include_rows]

                            ic_lst.append(include_ic.mean(axis=0))
                            tfidf_lst.append(include_tfidf.mean(axis=0))
                            classification_lst.append(max_feature_cl)
                    else:
                        pass

                    max_corr_lst.append(max_corr)
                    idx_lst.append(max_idx)
                    feature_lst.append(max_features)
                    max_pval_lst.append(max_pval)
                    segments_lst.append(segments)

                    method_slst = [f"{model}_{dset_name}_{method}"] * n_seg
                    method_lst.append(method_slst)
                    seg_sol_slst = [f"{file_i+3}"] * n_seg
                    seg_sol_lst.append(seg_sol_slst)
                    # ic_lst.append(0)
                    # tfidf_lst.append(0)

    max_corr_lst = np.hstack(max_corr_lst)
    idx_lst = np.hstack(idx_lst)
    feature_lst = np.hstack(feature_lst)
    max_pval_lst = np.hstack(max_pval_lst)
    segments_lst = np.hstack(segments_lst)
    seg_sol_lst = np.hstack(seg_sol_lst)
    method_lst = np.hstack(method_lst)
    ic_lst = np.hstack(ic_lst)
    tfidf_lst = np.hstack(tfidf_lst)
    classification_lst = np.hstack(classification_lst)

    data_df = pd.DataFrame()
    data_df["method"] = method_lst
    data_df["segment"] = segments_lst
    data_df["segment_solution"] = seg_sol_lst
    # data_df["segment"] = data_df["segment"].astype(str)
    data_df["max_corr"] = max_corr_lst
    data_df["pvalue"] = max_pval_lst
    data_df["corr_idx"] = idx_lst
    data_df["features"] = feature_lst
    data_df["information_content"] = ic_lst
    data_df["tfidf"] = tfidf_lst
    data_df["classification"] = classification_lst

    data_df.to_csv(op.join(output_dir, "performance.tsv"), sep="\t")

    return data_df


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
    # Define Paths
    # =============
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    templates_dir = op.join(project_dir, "data", "templates")
    hcp_gradient_dir = op.join(project_dir, "results", "hcp_gradient")
    gradient_segmentation_dir = op.join(project_dir, "results", "gradient_segmentation")
    gradient_decoding_dir = op.join(project_dir, "results", "gradient_decoding")
    decoding_performance_dir = op.join(project_dir, "results", "decoding_performance")
    # decoding_results_dir = op.join(project_dir, "results", "decoding_results")
    """
    # Run Workflow
    # =============
    # 1. Functional Connectivity Gradient
    print("1. Functional Connectivity Gradient", flush=True)
    principal_gradient_fn = op.join(hcp_gradient_dir, "principal_gradient.npy")
    if not op.isfile(principal_gradient_fn):
        principal_gradient = hcp_gradient(data_dir, templates_dir, principal_gradient_fn)
    else:
        print("\tGradient file exists. Loading principal gradient...", flush=True)
        principal_gradient = np.load(principal_gradient_fn)

    # 2. Segmentation and Gradient Maps
    n_segments = 30
    print("2. Segmentation and Gradient Maps", flush=True)
    grad_seg_fn = op.join(gradient_segmentation_dir, "grad_segments.pkl")
    if not op.isfile(grad_seg_fn):
        grad_seg_dict = gradient_segmentation(principal_gradient, grad_seg_fn, n_segments)
    else:
        print("\tGradient dict exists. Loading segmented gradient...", flush=True)
        grad_segments_file = open(grad_seg_fn, "rb")
        grad_seg_dict = pickle.load(grad_segments_file)

    # 3. Meta-Analytic Functional Decoding
    print("3. Meta-Analytic Functional Decoding", flush=True)
    gradient_decoding(
        data_dir,
        gradient_decoding_dir,
        grad_seg_dict,
        n_cores,
    )
    """
    # 4. Performance of Decoding Strategies
    print("4. Performance of Decoding Strategies", flush=True)
    # grad_seg_fn = op.join(gradient_segmentation_dir, "grad_segments.pkl")
    # if not op.isfile(grad_seg_fn):
    performance_df = decoding_performance(
        data_dir, gradient_decoding_dir, decoding_performance_dir
    )

    # 5. Visualization of the Decoded Maps
    # print("5. Visualization of the Decoded Maps", flush=True)
    # decoding_results(performance_df)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
