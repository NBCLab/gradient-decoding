"""Workflow for running the grdient-decoding analyses"""
import gzip
import os
import os.path as op
import pickle

import mapalign
import nibabel as nib
import numpy as np
from brainspace.gradient import GradientMaps
from neuromaps.datasets import fetch_annotation, fetch_fslr
from nibabel import GiftiImage
from nibabel.gifti import GiftiDataArray
from nimare.dataset import Dataset
from nimare.decode.continuous import CorrelationDecoder
from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma import mkda
from surfplot.utils import add_fslr_medial_wall

import utils
from segmentation import (
    KDESegmentation,
    KMeansSegmentation,
    PCTLSegmentation,
    compare_segmentations,
    gradient_to_maps,
)


def hcp_gradient(data_dir, template_dir, output_dir):
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

    print("Reading connenctivity matrix and apply Fisher's z-to-r transform...", flush=True)
    dcon_img = nib.load(
        op.join(data_dir, "hcp", "HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii")
    )
    dcon_mtx = np.tanh(dcon_img.get_fdata())  # 91,282 X 91,282 grayordinates
    dcon_mtx = dcon_mtx.astype("float32")

    del dcon_img

    print("Applying diffusion map embedding...", flush=True)
    # Calculate affinity matrix
    dcon_mtx = utils.affinity(dcon_mtx, 90)

    gradients, lambdas = mapalign.embed.compute_diffusion_map(
        dcon_mtx, alpha=0.5, return_result=True, overwrite=True
    )
    pickle.dump(lambdas, open(op.join(output_dir, "lambdas.p"), "wb"))
    pickle.dump(gradients, open(op.join(output_dir, "gradients.p"), "wb"))

    utils.plot_dm_results(lambdas, output_dir)

    del dcon_mtx

    print("Exporting gradient to nii and gii files...", flush=True)
    # Load subcortical volume
    subcortical_fn = op.join(template_dir, "rois-subcortical_mni152_mask.nii.gz")
    subcort_img = nib.load(subcortical_fn)
    subcort_dat = subcort_img.get_fdata()
    subcort_dat_idxs = np.nonzero(subcort_dat)[0]
    n_subcort_vox = len(subcort_dat_idxs)

    n_gradients = gradients.shape[1]
    for i in range(n_gradients):
        # Exclude 31,870 voxels form subcortical structures as represented in volumetric space
        # = 59,412 excluding the medial wall
        subcort_grads = gradients[gradients.shape[0] - n_subcort_vox :, i]
        gradients = gradients[: gradients.shape[0] - n_subcort_vox, i]

        # Add the medial wall: 32,492 X 32,492 grayordinates = 64,984, for visualization purposes
        # Get left and rigth hemisphere gradient scores, and insert 0's where medial wall is
        grad_map_full = add_fslr_medial_wall(gradients, split=False)
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
        new_subcort_dat[subcort_dat_idxs] = subcort_grads
        new_subcort_img = nib.Nifti1Image(new_subcort_dat, subcort_img.affine, subcort_img.header)
        new_subcort_img.to_filename(subcort_grads_fn)

        # Write cortical gradient to Gifti file
        nib.save(grad_img_lh, gradients_lh_fn)
        nib.save(grad_img_rh, gradients_rh_fn)

    principal_gradient_fn = op.join(hcp_gradient_dir, "principal_gradient.npz")
    principal_gradient = gradients[gradients.shape[0] - n_subcort_vox :, 0]
    np.savez(principal_gradient_fn, principal_gradient)

    return principal_gradient


def gradient_segmentation(gradient, output_dir):
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
    # 2.1. Segment the gradient into k ≥ 3 segments.
    n_segments = 5

    # Percentile Segmentation
    print("Running Percentile Segmentation...")
    percent_fn = op.join(output_dir, "percentile", "percentile_results.pkl")
    if not op.isfile(percent_fn):
        pctl_method = PCTLSegmentation(percent_fn, n_segments)
        percent_dict = pctl_method.fit(gradient)
    else:
        percent_file = open(percent_fn, "rb")
        percent_dict = pickle.load(percent_file)
        percent_file.close()

    percent_segments, percent_labels, percent_peaks = (
        percent_dict["segments"],
        percent_dict["labels"],
        percent_dict["peaks"],
    )

    # K-Means
    print("Running K-Means Segmentation...")
    kmeans_fn = op.join(output_dir, "kmeans", "kmeans_results.pkl")
    if not op.isfile(kmeans_fn):
        kmeans_method = KMeansSegmentation(kmeans_fn, n_segments)
        kmeans_dict = kmeans_method.fit(gradient)
    else:
        kmeans_file = open(kmeans_fn, "rb")
        kmeans_dict = pickle.load(kmeans_file)
        kmeans_file.close()

    kmeans_segments, kmeans_labels, kmeans_peaks = (
        kmeans_dict["segments"],
        kmeans_dict["labels"],
        kmeans_dict["peaks"],
    )

    # KDE
    print("Running KDE Segmentation...")
    kde_fn = op.join(output_dir, "kde", "kde_results.pkl")
    if not op.isfile(kde_fn):
        kde_method = KDESegmentation(kde_fn, n_segments)
        kde_dict = kde_method.fit(gradient)
    else:
        kde_file = open(kde_fn, "rb")
        kde_dict = pickle.load(kde_file)
        kde_file.close()

    kde_segments, kde_labels, kde_peaks = (
        kde_dict["segments"],
        kde_dict["labels"],
        kde_dict["peaks"],
    )

    assert len(percent_labels) == n_segments
    assert len(kmeans_labels) == n_segments
    assert len(kde_labels) == n_segments

    # Silhouette measures
    print("Calculating Silhouette measures...")
    silhouette_df_fn = op.join(output_dir, "silhouette_scores.csv")
    if not op.isfile(silhouette_df_fn):
        compare_segmentations(
            gradient, percent_labels, kmeans_labels, kde_labels, silhouette_df_fn
        )

    # 2.2. Transform KDE segmented gradient maps to activation maps.
    print("Transforming segmented gradient maps to activation maps...")
    # Percentile
    percent_seg_fn = op.join(output_dir, "percentile", "percentile_grad_segments_z.pkl")
    if not op.isfile(percent_seg_fn):
        percent_seg_dict = gradient_to_maps(
            "Percentile", percent_segments, percent_peaks, percent_seg_fn
        )
    else:
        percent_seg_file = open(percent_seg_fn, "rb")
        percent_seg_dict = pickle.load(percent_seg_file)
        percent_seg_file.close()

    percent_grad_segments_z = percent_seg_dict["grad_segments_z"]

    # KMeans
    kmeans_seg_fn = op.join(output_dir, "kmeans", "kmeans_grad_segments_z.pkl")
    if not op.isfile(kmeans_seg_fn):
        kmeans_seg_dict = gradient_to_maps("KMeans", kmeans_segments, kmeans_peaks, kmeans_seg_fn)
    else:
        kmeans_seg_file = open(kmeans_seg_fn, "rb")
        kmeans_seg_dict = pickle.load(kmeans_seg_file)
        kmeans_seg_file.close()

    kmeans_grad_segments_z = kmeans_seg_dict["grad_segments_z"]

    # KDE
    kde_seg_fn = op.join(output_dir, "kde", "kde_grad_segments_z.pkl")
    if not op.isfile(kde_seg_fn):
        kde_seg_dict = gradient_to_maps("KDE", kde_segments, kde_peaks, kde_seg_fn)
    else:
        kde_seg_file = open(kde_seg_fn, "rb")
        kde_seg_dict = pickle.load(kde_seg_file)
        kde_seg_file.close()

    kde_grad_segments_z = kde_seg_dict["grad_segments_z"]

    return None


def gradient_decoding(data_dir):
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
    output_dir = op.join(data_dir, "meta-analysis")
    os.makedirs(output_dir, exist_ok=True)
    datasets = ["neurosynth", "neuroquery"]

    for dataset in datasets:
        dset_fn = os.path.join(output_dir, f"{dataset}_dataset.pkl.gz")
        if not os.path.isfile(dset_fn):
            if dataset == "neurosynth":
                files = fetch_neurosynth(
                    data_dir=output_dir,
                    version="7",
                    overwrite=False,
                    source="abstract",
                    vocab="terms",
                )
            elif dataset == "neuroquery":
                files = fetch_neuroquery(
                    data_dir=output_dir,
                    version="1",
                    overwrite=False,
                    source="combined",
                    vocab="neuroquery7547",
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
            dset = Dataset.load(dset_fn)

        # Check whether dset contain text for LDA-based models
        if dataset == "neurosynth":
            if "abstract" in dset.texts:
                # Download Neurosynth abstract
                print("Dataset contains abstract")
            else:
                print("Downloading abstract to dset")
                dset = download_abstracts(dset, "jpera054@fiu.edu")
                dset.save(dset_fn)
        elif dataset == "neuroquery":
            if "title_keywords_abstract_body" in dset.texts:
                print("Dataset contains title_keywords_abstract_body")
            else:
                print("Downloading title_keywords_abstract_body to dset")
                # dset = download_abstracts(dset, "jpera054@fiu.edu")
                # dset.save(dset_fn)

        if dset.basepath is None:
            basepath_path = os.path.join(output_dir, f"{dataset}_basepath")
            os.makedirs(basepath_path, exist_ok=True)
            dset.update_path(basepath_path)

        print(dset.texts)
        # print(dset.texts["abstract"])
        # Term-based meta-analysis
        """
        print("Performing term-based meta-analysis...")
        term_based_decoder_fn = os.path.join(output_dir, f"term-based_{dataset}_decoder.pkl.gz")
        if not op.isfile(term_based_decoder_fn):
            decoder = CorrelationDecoder(
                frequency_threshold=0.001,
                meta_estimator=mkda.MKDAChi2,
                feature_group="terms_abstract_tfidf",
                target_image="z_desc-specificity",
            )
            decoder.fit(dset)
            decoder.save(term_based_decoder_fn, compress=True)
        else:
            decoder_file = gzip.open(term_based_decoder_fn, "rb")
            decoder = pickle.load(decoder_file)
    
        term_based_meta_maps = decoder.images_
        print(term_based_meta_maps)
        """

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


if __name__ == "__main__":
    # Define Paths
    # =============
    # project_dir = "/home/data/nbc/misc-projects/Peraza_GradientDecoding"
    project_dir = "/Users/jperaza/Documents/GitHub/gradient-decoding"
    templates_dir = op.join(project_dir, "data", "templates")
    data_dir = op.join(project_dir, "data")
    hcp_gradient_dir = op.join(project_dir, "results", "hcp_gradient")
    gradient_segmentation_dir = op.join(project_dir, "results", "gradient_segmentation")

    # Run Workflow
    # =============
    # 1. Functional Connectivity Gradient
    principal_gradient_fn = op.join(hcp_gradient_dir, "principal_gradient.npz")
    if not op.isfile(principal_gradient_fn):
        principal_gradient = hcp_gradient(data_dir, templates_dir, hcp_gradient_dir)
    else:
        print("Gradient dictionary file exists. Loading dictionary...", flush=True)
        principal_gradient = np.load(principal_gradient_fn)

    print(principal_gradient.shape, flush=True)
    print(principal_gradient, flush=True)

    # 2. Segmentation and Gradient Maps
    # grad_maps_z_fn = op.join(gradient_segmentation_dir, "grad_maps_z.npy")
    # if not op.isfile(grad_maps_z_fn):
    gradient_segmentation(principal_gradient, gradient_segmentation_dir)
    # else:
    #    grad_maps_z = np.load(grad_maps_z_fn)

    # 3. Meta-Analytic Functional Decoding
    gradient_decoding(data_dir)

    # 4. Performance of Decoding Strategies
    decoding_performance()

    # 5. Visualization of the Decoded Maps
    decoding_results()
