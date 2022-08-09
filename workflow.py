"""Workflow for running the grdient-decoding analyses"""
import argparse
import gzip
import os
import os.path as op
import pickle

import mapalign
import nibabel as nib
import numpy as np
from brainspace.gradient import GradientMaps
from nibabel import GiftiImage
from nibabel.gifti import GiftiDataArray
from nimare.dataset import Dataset
from nimare.decode.continuous import CorrelationDecoder
from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma import mkda
from surfplot.utils import add_fslr_medial_wall

import utils
from decoding import annotate_lda
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
    if not op.isfile(silhouette_df_fn):
        print("\tCalculating Silhouette measures...", flush=True)
        compare_segmentations(
            gradient, percent_labels, kmeans_labels, kde_labels, silhouette_df_fn
        )

    return grad_seg_dict


def gradient_decoding(
    data_dir, output_dir, percent_grad_segments, kmeans_grad_segments, kde_grad_segments, n_cores
):
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
    data_dir = op.join(data_dir, "meta-analysis")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    datasets = ["neurosynth", "neuroquery"]
    for dataset in datasets:
        dset_fn = os.path.join(data_dir, f"{dataset}_dataset.pkl.gz")
        if not os.path.isfile(dset_fn):
            print(f"\tFetching {dataset} database...", flush=True)
            if dataset == "neurosynth":
                files = fetch_neurosynth(
                    data_dir=data_dir,
                    version="7",
                    overwrite=False,
                    source="abstract",
                    vocab="terms",
                )
            elif dataset == "neuroquery":
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
            print(f"\tLoading {dataset} database...", flush=True)
            dset = Dataset.load(dset_fn)

        # Check whether dset contain text for LDA-based models
        if dataset == "neurosynth":
            if "abstract" in dset.texts:
                # Download Neurosynth abstract
                print(f"\t\tDataset {dataset} contains abstract", flush=True)
            else:
                print(f"\t\tDownloading abstract to {dataset} dset", flush=True)
                dset = download_abstracts(dset, "jpera054@fiu.edu")
                dset.save(dset_fn)
        elif dataset == "neuroquery":
            # LDA model will be run on word_counts, so the text is not needed
            pass

        if dset.basepath is None:
            basepath_path = op.join(data_dir, f"{dataset}_basepath")
            os.makedirs(basepath_path, exist_ok=True)
            dset.update_path(basepath_path)

        # Term-based meta-analysis
        if dataset == "neurosynth":
            feature_group = "terms_abstract_tfidf"
        elif dataset == "neuroquery":
            feature_group = "neuroquery6308_combined_tfidf"

        term_based_decoder_fn = op.join(output_dir, f"term-based_{dataset}_decoder.pkl.gz")
        if not op.isfile(term_based_decoder_fn):
            print(f"\tPerforming Term-based meta-analysis on {dataset}...", flush=True)
            decoder = CorrelationDecoder(
                frequency_threshold=0.001,
                meta_estimator=mkda.MKDAChi2,
                feature_group=feature_group,
                target_image="z_desc-specificity",
                n_cores=n_cores,
            )
            decoder.fit(dset)
            decoder.save(term_based_decoder_fn, compress=True)
        else:
            print(f"\tLoading Term-based meta-analytic maps from {dataset}...", flush=True)
            decoder_file = gzip.open(term_based_decoder_fn, "rb")
            decoder = pickle.load(decoder_file)

        # LDA-based meta-analysis
        n_topics = 200
        feature_group = f"LDA{n_topics}"
        lda_based_decoder_fn = op.join(output_dir, f"lda-based_{dataset}_decoder.pkl.gz")
        if not op.isfile(lda_based_decoder_fn):
            print(f"\tRunning LDA model on {dataset}...", flush=True)
            # n_cores=1 for LDA. See: https://github.com/scikit-learn/scikit-learn/issues/8943
            dset = annotate_lda(dset, dataset, data_dir, n_topics=n_topics, n_cores=1)
            dset.save(dset_fn)

            print(f"\tPerforming LDA-based meta-analysis on {dataset}...", flush=True)
            lda_decoder = CorrelationDecoder(
                frequency_threshold=0.05,
                meta_estimator=mkda.MKDAChi2,
                feature_group=feature_group,
                target_image="z_desc-specificity",
                n_cores=n_cores,
            )
            lda_decoder.fit(dset)
            lda_decoder.save(lda_based_decoder_fn, compress=True)
        else:
            print(f"\tLoading LDA-based meta-analytic maps from {dataset}...", flush=True)
            lda_decoder_file = gzip.open(lda_based_decoder_fn, "rb")
            lda_decoder = pickle.load(lda_decoder_file)

        # GCLDA-based meta-analysis

    return None


def decoding_performance(data_dir):
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

    return None


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
    # decoding_performance_dir = op.join(project_dir, "results", "decoding_performance")
    # decoding_results_dir = op.join(project_dir, "results", "decoding_results")

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

    percent_grad_segments = grad_seg_dict["percentile_grad_segments"]
    kmeans_grad_segments = grad_seg_dict["kmeans_grad_segments"]
    kde_grad_segments = grad_seg_dict["kde_grad_segments"]

    # 3. Meta-Analytic Functional Decoding
    print("3. Meta-Analytic Functional Decoding", flush=True)
    gradient_decoding(
        data_dir,
        gradient_decoding_dir,
        percent_grad_segments,
        kmeans_grad_segments,
        kde_grad_segments,
        n_cores,
    )

    # 4. Performance of Decoding Strategies
    # print("4. Performance of Decoding Strategies", flush=True)
    # decoding_performance()

    # 5. Visualization of the Decoded Maps
    # print("5. Visualization of the Decoded Maps", flush=True)
    # decoding_results()


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
