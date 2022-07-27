"""Miscellaneous functions used for analyses."""
import os.path as op

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity


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
    cmap = plt.cm.get_cmap("jet")
    markersize = 15

    print("Number of optimal bins:", len(max_vals), flush=True)
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
        op.join(output_dir, f"bins-segment-{n_segment}.png"), bbox_inches="tight", pad_inches=0
    )


def percent_segmentation(gradient, n_segments):
    MIN_N_SEGMENTS = 3  # Minimum number of segments

    percent_segments = []
    percent_labels = []
    for n_segment in range(MIN_N_SEGMENTS, n_segments + MIN_N_SEGMENTS):
        step = int(100 / n_segment)
        bins = list(zip(np.arange(0, 100, step), np.arange(step, 100 + step, step)))

        labels_arr = np.zeros_like(gradient)
        gradient_maps = []
        for i, b in enumerate(bins):
            min_, max_ = np.percentile(gradient[np.where(gradient)], b)
            # Threshold gradient map based on bin, but don’t binarize.
            thresh_arr = gradient.copy()
            thresh_arr[thresh_arr < min_] = 0
            thresh_arr[thresh_arr > max_] = 0
            gradient_maps.append(thresh_arr)

            labels_arr[np.where(thresh_arr != 0)] = i + 1

        percent_segments.append(gradient_maps)
        percent_labels.append(labels_arr)

    return percent_segments, percent_labels


def kmeans_segmentation(gradient, n_segments):
    MIN_N_SEGMENTS = 3  # Minimum number of segments

    kmeans_labels = []
    for n_segment in range(MIN_N_SEGMENTS, n_segments + MIN_N_SEGMENTS):
        kmeans_model = KMeans(
            n_clusters=n_segment,
            init="k-means++",
            n_init=10,
            random_state=0,
            algorithm="elkan",
        ).fit(gradient)

        kmeans_labels.append(kmeans_model.labels_)

    return kmeans_labels


def kde_segmentation(gradient, output_dir, n_segments):

    MIN_N_SEGMENTS = 3  # Minimum number of segments
    bw_used = int(gradient.max()) / 2

    kde_segments = []
    kde_labels = []
    for n_segment in range(MIN_N_SEGMENTS, n_segments + MIN_N_SEGMENTS):
        n_kde_segments = 0
        while n_segment != n_kde_segments:
            kde = KernelDensity(kernel="gaussian", bandwidth=bw_used).fit(gradient.reshape(-1, 1))
            x_samples = np.linspace(gradient.min(), gradient.max(), num=gradient.shape[0])
            y_densities = kde.score_samples(x_samples.reshape(-1, 1))

            min_vals, max_vals = (
                argrelextrema(y_densities, np.less)[0],
                argrelextrema(y_densities, np.greater)[0],
            )
            n_kde_segments = len(max_vals)

        print(f"Found {n_segment} number of segments for bin width {bw_used}")
        plot_kde_segmentation(n_segment, x_samples, y_densities, min_vals, max_vals, output_dir)

        print(min_vals)
        print(max_vals)
        labels_arr = np.zeros_like(gradient)
        gradient_maps = []
        for i in range(len(max_vals)):
            min_, max_ = min_vals[i], min_vals[i + 1]
            # Threshold gradient map based on bin, but don’t binarize.
            thresh_arr = gradient.copy()
            thresh_arr[thresh_arr < min_] = 0
            thresh_arr[thresh_arr > max_] = 0
            gradient_maps.append(thresh_arr)

            labels_arr[np.where(thresh_arr != 0)] = i + 1

        kde_segments.append(gradient_maps)
        kde_labels.append(labels_arr)

    return kde_segments, kde_labels


def compare_segmentations():

    return None
