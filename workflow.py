"""Workflow for running the grdient-decoding analyses"""
import os.path as op
import pickle

import nibabel as nib
import numpy as np
from brainspace.gradient import GradientMaps
from surfplot.utils import add_fslr_medial_wall

import utils


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
    print("Reading connenctivity matrix and apply Fisher's z-to-r transform...", flush=True)
    dcon_img = nib.load(
        op.join(data_dir, "hcp", "HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii")
    )
    dcon_mtx = np.tanh(dcon_img.get_fdata())  # 91,282 X 91,282 grayordinates
    dcon_mtx = dcon_mtx.astype("float32")

    del dcon_img

    print("Applying diffusion map embedding...", flush=True)
    gm = GradientMaps(n_components=10, random_state=0, kernel="cosine", approach="dm")
    gm.fit(dcon_mtx, sparsity=0.9, n_iter=10)
    lambdas = gm.lambdas_
    gradients = gm.gradients_
    pickle.dump(lambdas, open(op.join(output_dir, "lambdas_temp.p"), "wb"))
    pickle.dump(gradients, open(op.join(output_dir, "gradients_temp.p"), "wb"))

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
        gradients_lh, gradients_rh = add_fslr_medial_wall(gradients, split=True)

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
        nib.save(gradients_lh, gradients_lh_fn)
        nib.save(gradients_rh, gradients_rh_fn)

    principal_gradient_fn = op.join(hcp_gradient_dir, "principal_gradient.npz")
    principal_gradient = gradients[gradients.shape[0] - n_subcort_vox :, 0]
    np.savez(principal_gradient_fn, principal_gradient)

    return principal_gradient


def gradient_segmentation():
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

    return None


def gradient_decoding():
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

    return None


def decoding_performance():
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
    project_dir = "/home/data/nbc/misc-projects/Peraza_GradientDecoding"
    templates_dir = op.join(project_dir, "data", "templates")
    data_dir = op.join(project_dir, "data")
    hcp_gradient_dir = op.join(project_dir, "results", "hcp_gradient")

    # Run Workflow
    # =============
    principal_gradient_fn = op.join(hcp_gradient_dir, "principal_gradient.npz")
    if not op.isfile(principal_gradient_fn):
        principal_gradient = hcp_gradient(data_dir, templates_dir, hcp_gradient_dir)
    else:
        print("Gradient dictionary file exists. Loading dictionary...", flush=True)
        principal_gradient = np.load(principal_gradient_fn)

    print(principal_gradient.shape, flush=True)
    print(principal_gradient, flush=True)

    gradient_segmentation()
    gradient_decoding()
    decoding_performance()
    decoding_results()
