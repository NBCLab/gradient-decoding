"""Miscellaneous functions used for analyses."""
import logging
import os
import os.path as op

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from neuromaps.datasets import fetch_atlas
from nibabel import GiftiImage
from nibabel.gifti import GiftiDataArray
from sklearn.metrics import pairwise_distances
from surfplot.utils import add_fslr_medial_wall

LGR = logging.getLogger(__name__)


def rm_fslr_medial_wall(data_lh, data_rh, neuromaps_dir, join=True):
    """Remove medial wall from data in fsLR space
    Data in 32k fs_LR space (e.g., Human Connectome Project data) often in
    GIFTI format include the medial wall in their data arrays, which results
    in a total of 64984 vertices across hemispheres. This function removes
    the medial wall vertices to produce a data array with the full 59412 vertices,
    which is used to perform functional decoding.

    This function was adapted from :func:`surfplot.utils.add_fslr_medial_wall`.

    Parameters
    ----------
    data : numpy.ndarray
        Surface vertices. Must have exactly 32492 vertices per hemisphere.
    join : bool
        Return left and right hemipsheres in the same arrays. Default: True
    Returns
    -------
    numpy.ndarray
        Vertices with medial wall excluded (59412 vertices total)
    Raises
    ------
    ValueError
        `data` has the incorrect number of vertices (59412 or 64984 only
        accepted)
    """
    assert data_lh.shape[0] == 32492
    assert data_rh.shape[0] == 32492

    atlas = fetch_atlas("fsLR", "32k", data_dir=neuromaps_dir, verbose=0)
    medial_lh, medial_rh = atlas["medial"]
    wall_lh = nib.load(medial_lh).agg_data()
    wall_rh = nib.load(medial_rh).agg_data()

    data_lh = data_lh[np.where(wall_lh != 0)]
    data_rh = data_rh[np.where(wall_rh != 0)]

    if join:
        data = np.hstack((data_lh, data_rh))
        assert data.shape[0] == 59412
        return data
    else:
        return data_lh, data_rh


def zero_fslr_medial_wall(data_lh, data_rh, neuromaps_dir):
    """Remove medial wall from data in fsLR space"""

    atlas = fetch_atlas("fsLR", "32k", data_dir=neuromaps_dir, verbose=0)
    medial_lh, medial_rh = atlas["medial"]
    medial_arr_lh = nib.load(medial_lh).agg_data()
    medial_arr_rh = nib.load(medial_rh).agg_data()

    data_arr_lh = data_lh.agg_data()
    data_arr_rh = data_rh.agg_data()
    data_arr_lh[np.where(medial_arr_lh == 0)] = 0
    data_arr_rh[np.where(medial_arr_rh == 0)] = 0

    data_lh.remove_gifti_data_array(0)
    data_rh.remove_gifti_data_array(0)
    data_lh.add_gifti_data_array(GiftiDataArray(data_arr_lh))
    data_rh.add_gifti_data_array(GiftiDataArray(data_arr_rh))

    return data_lh, data_rh


def affinity(matrix, sparsity):
    # Generate percentile thresholds for 90th percentile
    perc = np.array([np.percentile(x, sparsity) for x in matrix])

    # Threshold each row of the matrix by setting values below 90th percentile to 0
    for i in range(matrix.shape[0]):
        matrix[i, matrix[i, :] < perc[i]] = 0
    matrix[matrix < 0] = 0

    # Now we are dealing with sparse vectors. Cosine similarity is used as affinity metric
    matrix = 1 - pairwise_distances(matrix, metric="cosine")

    return matrix


def plot_dm_results(lambdas, output_dir):
    _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    ax1.set_xlabel("Component Nb")
    ax1.set_xlabel("Optimal Components")

    ax1.set_ylabel("Explained Variance Ratio")
    ax2.set_ylabel("Difference in Explained Variance Ratio")
    ax1.scatter(range(1, lambdas.size + 1, 1), (100 * lambdas) / lambdas.sum())
    gm_lambdas_diff = 100 * (lambdas[:-1] - lambdas[1:]) / lambdas.sum()
    ax2.scatter(range(1, lambdas.size, 1), gm_lambdas_diff)

    plt.savefig(op.join(output_dir, "lambdas.png"))


def plot_kde_segmentation(n_segment, x_samples, y_densities, min_vals, max_vals, output_dir):
    x_min, x_max = x_samples.min(), x_samples.max()
    y_min = y_densities.min()
    cmap = plt.cm.get_cmap("viridis")
    markersize = 15

    _, axs = plt.subplots(figsize=(15, 5))

    points = np.array([x_samples, y_densities]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(x_min, x_max)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(x_samples)
    lc.set_linewidth(2)
    axs.add_collection(lc)
    axs.set_axis_off()
    axs.set_xlim([x_min, x_max])

    npts = len(x_samples)
    for i in range(npts - 1):
        plt.fill_between(
            [x_samples[i], x_samples[i + 1]],
            [y_densities[i], y_densities[i + 1]],
            y2=y_min,
            alpha=0.5,
            color=cmap(norm(x_samples[i])),
        )

    plt.plot(
        x_samples[max_vals],
        y_densities[max_vals],
        markersize=markersize,
        linestyle="",
        marker="o",
        markerfacecolor="gray",
        markeredgecolor="black",
    )
    plt.plot(
        x_samples[min_vals],
        y_densities[min_vals],
        markersize=markersize,
        linestyle="",
        marker="s",
        markerfacecolor="k",
        markeredgecolor="k",
    )
    plt.vlines(
        x=x_samples[min_vals],
        ymin=y_min,
        ymax=y_densities[min_vals],
        colors="k",
        ls="--",
        lw=2,
    )
    """
    sns.stripplot(
        x=x,
        y=y,
        edgecolor="white",
        dodge=False,
        size=3,
        cmap=cmap,
        jitter=1,
        zorder=0,
        orient="h",
    )
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        op.join(output_dir, "kde-segment-{:02d}.png".format(n_segment)),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close("all")


def _gradient_to_gifti(gradients, subcort_img, principal_gradient_fn, output_dir):
    """Convert gradient to gifti format"""
    # Add the medial wall: 32,492 X 32,492 grayordinates = 64,984, for visualization purposes
    # Get left and rigth hemisphere gradient scores, and insert 0's where medial wall is
    full_vertices = 64984
    hemi_vertices = full_vertices // 2

    subcort_dat = subcort_img.get_fdata()
    subcort_mask = subcort_dat != 0
    n_subcort_vox = np.where(subcort_mask)[0].shape[0]

    n_gradients = gradients.shape[1]
    for i in range(n_gradients):
        cort_grads = gradients[: gradients.shape[0] - n_subcort_vox, i]

        if i == 0:
            # Save principal gradient
            principal_gradient = cort_grads.copy()
            np.save(principal_gradient_fn, principal_gradient)

        grad_map_full = add_fslr_medial_wall(cort_grads, split=False)
        gradients_lh, gradients_rh = grad_map_full[:hemi_vertices], grad_map_full[hemi_vertices:]

        grad_img_lh = GiftiImage()
        grad_img_rh = GiftiImage()
        grad_img_lh.add_gifti_data_array(GiftiDataArray(gradients_lh))
        grad_img_rh.add_gifti_data_array(GiftiDataArray(gradients_rh))

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

        # Write cortical gradient to Gifti file
        nib.save(grad_img_lh, gradients_lh_fn)
        nib.save(grad_img_rh, gradients_rh_fn)

    return principal_gradient


def _gradient_to_nifti(gradients, subcort_img, output_dir):
    """Convert sub-cortical gradient to nifti format"""
    subcort_dat = subcort_img.get_fdata()
    subcort_mask = subcort_dat != 0
    n_subcort_vox = np.where(subcort_mask)[0].shape[0]

    n_gradients = gradients.shape[1]
    for i in range(n_gradients):
        # Exclude 31,870 voxels form subcortical structures as represented in volumetric space
        # = 59,412 excluding the medial wall
        subcort_grads = gradients[gradients.shape[0] - n_subcort_vox :, i]

        subcort_grads_fn = op.join(
            output_dir,
            "source-jperaza2022_desc-fcG{:02d}_space-MNI152_den-2mm_feature.nii.gz".format(i),
        )

        # Write subcortical gradient to Nifti file
        new_subcort_dat = np.zeros_like(subcort_dat)
        new_subcort_dat[subcort_mask] = subcort_grads
        new_subcort_img = nib.Nifti1Image(new_subcort_dat, subcort_img.affine, subcort_img.header)
        new_subcort_img.to_filename(subcort_grads_fn)
