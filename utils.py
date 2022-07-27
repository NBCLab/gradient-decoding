"""Miscellaneous functions used for analyses."""
import os.path as op

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity


def rm_fslr_medial_wall(data_lh, data_rh, wall_lh, wall_rh, join=True):
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

    data_lh = data_lh[np.where(wall_lh != 0)]
    data_rh = data_rh[np.where(wall_rh != 0)]

    if join:
        data = np.hstack((data_lh, data_rh))
        assert data.shape[0] == 59412
        return data
    else:
        return data_lh, data_rh


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
    plt.savefig(
        op.join(output_dir, "bins-segment-{:02d}.png".format(n_segment)),
        bbox_inches="tight",
        pad_inches=0,
    )


def percent_segmentation(gradient, n_segments, min_n_segments=3):
    """KMeans-based segmentation.

    Thi method implements the original method described in (Margulies et al., 2016)
    in which the whole-brain gradient is segmented into equidistant gradient segments.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient vector.
    n_segments : int
        Total number of segments.
    min_n_segments : int
        Minimum number of segments.
    Returns
    -------
    kde_labels : list of numpy.ndarray
        Vertices labeled
    """
    percent_segments = []
    percent_labels = []
    for n_segment in range(min_n_segments, n_segments + min_n_segments):
        step = 100 / n_segment
        bins = list(zip(np.linspace(0, 100 - step, n_segment), np.linspace(step, 100, n_segment)))

        labels_arr = np.zeros_like(gradient)
        gradient_maps = []
        for i, bin in enumerate(bins):
            min_, max_ = np.percentile(gradient[np.where(gradient)], bin)
            # Threshold gradient map based on bin, but don’t binarize.
            thresh_arr = gradient.copy()
            thresh_arr[thresh_arr < min_] = 0
            thresh_arr[thresh_arr > max_] = 0
            gradient_maps.append(thresh_arr)

            labels_arr[np.where(thresh_arr != 0)] = i + 1

        percent_segments.append(gradient_maps)
        percent_labels.append(labels_arr)

    return percent_segments, percent_labels


def kmeans_segmentation(gradient, n_segments, min_n_segments=3):
    """KMeans-based segmentation.

    This method relied on 1D k-means clustering, which has previously
    been used to define clusters of functional connectivity matrices
    to establish a brain-wide parcellation.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient vector.
    n_segments : int
        Total number of segments.
    min_n_segments : int
        Minimum number of segments.
    Returns
    -------
    kde_labels : list of numpy.ndarray
        Vertices labeled
    """
    kmeans_labels = []
    for n_segment in range(min_n_segments, n_segments + min_n_segments):
        kmeans_model = KMeans(
            n_clusters=n_segment,
            init="k-means++",
            n_init=10,
            random_state=0,
            algorithm="elkan",
        ).fit(gradient.reshape(-1, 1))

        kmeans_labels.append(kmeans_model.labels_)

    return kmeans_labels


def kde_segmentation(
    gradient, output_dir, n_segments, min_n_segments=3, bandwidth=0.25, bw_step=0.001
):
    """KDE-based segmentation.

        This method identifies local minima of a Kernel Density Estimation (KDE)
        curve of the gradient axis, which are points with minimal numbers of vertices
        in their vicinity. The number of segments for the KDE method was modified by
        tuning the width parameter used to generate the KDE curve.
    0.095
        Parameters
        ----------
        gradient : numpy.ndarray
            Gradient vector.
        output_dir : str
        n_segments : int
            Total number of segments.
        min_n_segments : int
            Minimum number of segments. Default is 3.
        bandwidth : float
            Initialize bandwidth. Default is 0.25.
        bw_step : float
            Step to explore different bandwidths. Default is 0.001.
        Returns
        -------
        kde_segments : list of numpy.ndarray
            List with thresholded gradients maps.
        kde_labels : list of numpy.ndarray
            Vertices labeled
        Raises
        ------
        ValueError
            `bandwidth` reach a value less than zero. The algorithm did not
            converge with the current `bw_step`.

    """
    kde_segments = []
    kde_labels = []
    bw_used = []
    for n_segment in range(min_n_segments, n_segments + min_n_segments):
        n_kde_segments = 0
        print(n_segment)
        while n_segment != n_kde_segments:
            bandwidth -= bw_step
            if bandwidth < 0:
                raise ValueError(
                    f"Bandwidth = {bandwidth} has been reached. "
                    "Only positive values are allowed for bandwidth."
                )

            print(bandwidth)
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                gradient.reshape(-1, 1)
            )
            x_samples = np.linspace(gradient.min(), gradient.max(), num=gradient.shape[0])
            y_densities = kde.score_samples(x_samples.reshape(-1, 1))

            min_vals, max_vals = (
                argrelextrema(y_densities, np.less)[0],
                argrelextrema(y_densities, np.greater)[0],
            )
            n_kde_segments = len(max_vals)
            print(n_segment, n_kde_segments)

        bw_used.append(bandwidth)
        print(f"Found {n_segment} number of segments for bandwidth {bandwidth}")
        plot_kde_segmentation(n_segment, x_samples, y_densities, min_vals, max_vals, output_dir)

        labels_arr = np.zeros_like(gradient)
        gradient_maps = []
        for i in range(n_kde_segments):
            if i == 0:
                min_ = gradient.min()
                max_ = x_samples[min_vals[i]]
            elif i == n_kde_segments - 1:
                min_ = x_samples[min_vals[i - 1]]
                max_ = gradient.max()
            else:
                min_ = x_samples[min_vals[i - 1]]
                max_ = x_samples[min_vals[i]]

            # Threshold gradient map based on bin, but don’t binarize.
            thresh_arr = gradient.copy()
            thresh_arr[thresh_arr < min_] = 0
            thresh_arr[thresh_arr > max_] = 0
            gradient_maps.append(thresh_arr)

            labels_arr[np.where(thresh_arr != 0)] = i + 1

        kde_segments.append(gradient_maps)
        kde_labels.append(labels_arr)

    bw_used = np.hstack(bw_used)
    bw_used_fn = op.join(output_dir, "bandwidth_used.npy")
    np.save(bw_used_fn, bw_used)

    return kde_segments, kde_labels


def compare_segmentations():

    return None
