"""Extract information from HCP templates."""
import os
import os.path as op
import pickle
import warnings
from abc import ABCMeta

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel import GiftiImage
from nibabel.gifti import GiftiDataArray
from scipy.signal import argrelextrema
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity
from surfplot.utils import add_fslr_medial_wall
from tqdm import tqdm

import utils


class Segmentation(metaclass=ABCMeta):
    """Base class for segmentation methods."""

    def __init__(self):
        pass

    def fit(self, gradient):
        """Fit Segmentation to gradient."""
        segments, labels, boundaries, peaks = self._fit(gradient)

        # Save results to dictionary file
        segmentation_dict = {
            "segments": segments,
            "labels": labels,
            "boundaries": boundaries,
            "peaks": peaks,
        }

        os.makedirs(op.dirname(self.segmentation_fn), exist_ok=True)
        segmentation_file = open(self.segmentation_fn, "wb")
        pickle.dump(segmentation_dict, segmentation_file)
        segmentation_file.close()

        return segmentation_dict


class PCTLSegmentation(Segmentation):
    """Percentile-based segmentation.

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
    """

    def __init__(
        self,
        segmentation_fn,
        n_segments,
        min_n_segments=3,
    ):
        self.segmentation_fn = segmentation_fn
        self.n_segments = n_segments
        self.min_n_segments = min_n_segments

    def _fit(self, gradient):
        """Fit Segmentation to gradient.

         Parameters
        ----------
        gradient : numpy.ndarray
            Gradient vector.

        Returns
        -------
        segments : list of numpy.ndarray
            List with thresholded gradients maps.
        kde_labels : list of numpy.ndarray
            Vertices labeled.
        labels :
        boundaries :
        peaks :
        """
        segments = []
        labels = []
        boundaries = []
        peaks = []
        for n_segment in range(self.min_n_segments, self.n_segments + self.min_n_segments):
            step = 100 / n_segment
            bins = list(
                zip(np.linspace(0, 100 - step, n_segment), np.linspace(step, 100, n_segment))
            )

            gradient_maps = []
            labels_arr = np.zeros_like(gradient)
            map_bounds = []
            map_peaks = []
            map_bounds.append(gradient.min())
            for i, bin in enumerate(bins):
                # Get boundary points
                min_, max_ = np.percentile(gradient[np.where(gradient)], bin)

                # Threshold gradient map based on bin, but don’t binarize.
                thresh_arr = gradient.copy()
                thresh_arr[thresh_arr < min_] = 0
                thresh_arr[thresh_arr > max_] = 0
                gradient_maps.append(thresh_arr)

                non_zero_arr = np.where(thresh_arr != 0)
                labels_arr[non_zero_arr] = i
                map_bounds.append(max_)
                # Peak activation = median 50th percentile of the segment
                map_peaks.append(np.median(thresh_arr[non_zero_arr]))

            segments.append(gradient_maps)
            labels.append(labels_arr)
            boundaries.append(map_bounds)
            peaks.append(map_peaks)

        return segments, labels, boundaries, peaks


class KMeansSegmentation(Segmentation):
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
    """

    def __init__(
        self,
        segmentation_fn,
        n_segments,
        min_n_segments=3,
    ):
        self.segmentation_fn = segmentation_fn
        self.n_segments = n_segments
        self.min_n_segments = min_n_segments

    def _fit(self, gradient):
        """Fit Segmentation to gradient.

         Parameters
        ----------
        gradient : numpy.ndarray
            Gradient vector.

        Returns
        -------
        segments : list of numpy.ndarray
            List with thresholded gradients maps.
        kde_labels : list of numpy.ndarray
            Vertices labeled
        labels :
        boundaries :
        peaks :
        """
        segments = []
        labels = []
        boundaries = []
        peaks = []
        for n_segment in range(self.min_n_segments, self.n_segments + self.min_n_segments):
            kmeans_model = KMeans(
                n_clusters=n_segment,
                init="k-means++",
                n_init=10,
                random_state=0,
                algorithm="elkan",
            ).fit(gradient.reshape(-1, 1))

            # Get order mapper from map_peaks
            map_peaks = kmeans_model.cluster_centers_.flatten()
            order_idx = np.argsort(map_peaks)
            order_mapper = np.zeros_like(order_idx)
            order_mapper[order_idx] = np.arange(n_segment)

            # Reorder labels based on map_peaks order
            labels_arr = order_mapper[kmeans_model.labels_]

            gradient_maps = []
            map_bounds = []
            map_bounds.append(gradient.min())
            for i in range(n_segment):
                map_arr = np.zeros_like(gradient)
                map_arr[labels_arr == i] = gradient[labels_arr == i]
                gradient_maps.append(map_arr)

                map_bounds.append(map_arr.max())

            segments.append(gradient_maps)
            labels.append(labels_arr)
            boundaries.append(map_bounds)
            peaks.append(np.sort(map_peaks))

        return segments, labels, boundaries, peaks


class KDESegmentation(Segmentation):
    """KDE-based segmentation.

    This method identifies local minima of a Kernel Density Estimation (KDE)
    curve of the gradient axis, which are points with minimal numbers of vertices
    in their vicinity. The number of segments for the KDE method was modified by
    tuning the width parameter used to generate the KDE curve.

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
        Initialize bandwidth. Default is 0.26.
    bw_step : float
        Step to explore different bandwidths. Default is 0.001.
    Raises
    ------
    ValueError
        `bandwidth` reach a value less than zero. The algorithm did not
        converge with the current `bw_step`.
    """

    def __init__(
        self,
        segmentation_fn,
        n_segments,
        min_n_segments=3,
        bandwidth=0.26,
        bw_step=0.05,
    ):
        if bandwidth < bw_step:
            raise ValueError(
                f"bw_step = {bandwidth} should be smaller than initial bandwidth = {bandwidth}"
            )
        self.segmentation_fn = segmentation_fn
        self.n_segments = n_segments
        self.min_n_segments = min_n_segments
        self.bandwidth = bandwidth
        self.bw_step = bw_step

    def _fit(self, gradient):
        """Fit Segmentation to gradient.

         Parameters
        ----------
        gradient : numpy.ndarray
            Gradient vector.

        Returns
        -------
        segments : list of numpy.ndarray
            List with thresholded gradients maps.
        kde_labels : list of numpy.ndarray
            Vertices labeled
        labels :
        boundaries :
        peaks :

        """
        bandwidth_used_fn = op.join(op.dirname(self.segmentation_fn), "bandwidth_used.npy")
        os.makedirs(op.dirname(self.segmentation_fn), exist_ok=True)
        if not op.isfile(bandwidth_used_fn):
            wr_bandwidth_file = True
            bandwidths = np.arange(0, self.bandwidth + self.bw_step, self.bw_step)[::-1]
            bw_used = []
        else:
            bandwidths = np.load(bandwidth_used_fn)

        segments = []
        labels = []
        boundaries = []
        peaks = []
        bandwidth_i = 0
        for n_segment in tqdm(range(self.min_n_segments, self.n_segments + self.min_n_segments)):
            n_kde_segments = 0
            while n_segment != n_kde_segments:
                bandwidth = bandwidths[bandwidth_i]

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
                print(f"\t\t\t{round(bandwidth, 6)}: {n_segment}, {n_kde_segments}", flush=True)

                if n_segment < n_kde_segments:
                    new_bw_step = self.bw_step / 2
                    print(
                        "\t\t\t\tn_kde_segments cannot exceed n_segment. Resetting bandwidths "
                        f"array with a bandwidths smaller than {self.bw_step}, {new_bw_step}",
                        flush=True,
                    )
                    bandwidths = np.arange(0, bandwidth + self.bw_step, new_bw_step)[::-1]
                    self.bw_step = new_bw_step
                    bandwidth_i = 0  # Reset iterator
                else:
                    bandwidth_i += 1

            print(
                f"\t\t\tFound {n_segment} number of segments for bandwidth {round(bandwidth, 6)}",
                flush=True,
            )
            if wr_bandwidth_file:
                bw_used.append(bandwidth)

            utils.plot_kde_segmentation(
                n_segment,
                x_samples,
                y_densities,
                min_vals,
                max_vals,
                op.dirname(self.segmentation_fn),
            )

            gradient_maps = []
            labels_arr = np.zeros_like(gradient)
            map_bounds = []
            map_bounds.append(gradient.min())
            for i in range(n_kde_segments):
                # Get boundary points
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

                labels_arr[np.where(thresh_arr != 0)] = i
                map_bounds.append(max_)

            segments.append(gradient_maps)
            labels.append(labels_arr)
            boundaries.append(map_bounds)
            peaks.append(x_samples[max_vals])

        if wr_bandwidth_file:
            np.save(bandwidth_used_fn, bw_used)

        return segments, labels, boundaries, peaks


def compare_segmentations(gradient, percent_labels, kmeans_labels, kde_labels, silhouette_df_fn):
    n_segments = len(percent_labels)
    segment_sizes = [labels.max() + 1 for labels in percent_labels]
    silhouette_df = pd.DataFrame(index=segment_sizes)
    silhouette_df.index.name = "segment_sizes"

    percent_scores = np.zeros(n_segments)
    kmeans_scores = np.zeros(n_segments)
    kde_scores = np.zeros(n_segments)
    for segment_i in range(n_segments):
        percent_scores[segment_i] = silhouette_score(
            gradient.reshape(-1, 1), percent_labels[segment_i], metric="euclidean"
        )
        kmeans_scores[segment_i] = silhouette_score(
            gradient.reshape(-1, 1), kmeans_labels[segment_i], metric="euclidean"
        )
        kde_scores[segment_i] = silhouette_score(
            gradient.reshape(-1, 1), kde_labels[segment_i], metric="euclidean"
        )

    silhouette_df["percentile"] = percent_scores
    silhouette_df["kmeans"] = kmeans_scores
    silhouette_df["kde"] = kde_scores

    silhouette_df.to_csv(silhouette_df_fn)


def gradient_to_maps(method, segments, peaks, grad_seg_dict, output_dir):
    "Transform segmented gradient maps to normalized activation maps."
    full_vertices = 64984
    hemi_vertices = int(full_vertices / 2)

    segment_sizes = [len(segments) for segments in segments]
    grad_segments = []
    for seg_i, segment in enumerate(segments):
        grad_maps = []
        for map_i, grad_map in enumerate(segment):
            # Vertices located above the cluster_centers_ in the segment map
            # were translated relative to the maximum
            grad_map_peak = peaks[seg_i][map_i]
            vrtxs_to_translate = np.where((grad_map > grad_map_peak) & (grad_map != 0))
            grad_map[vrtxs_to_translate] = grad_map_peak - np.abs(grad_map[vrtxs_to_translate])

            # The resulting segment’s map was standardized into a z-score map
            # grad_map = abs(grad_map)
            # grad_map = zscore(grad_map)
            grad_maps.append(grad_map)

            grad_map_lh_fn = op.join(
                output_dir,
                "source-{}{:02d}_desc-{:02d}_space-fsLR_den-32k_hemi-L_feature.func.gii".format(
                    method, segment_sizes[seg_i], map_i
                ),
            )
            grad_map_rh_fn = op.join(
                output_dir,
                "source-{}{:02d}_desc-{:02d}_space-fsLR_den-32k_hemi-R_feature.func.gii".format(
                    method, segment_sizes[seg_i], map_i
                ),
            )

            if not (op.isfile(grad_map_lh_fn) and op.isfile(grad_map_rh_fn)):
                grad_map_full = add_fslr_medial_wall(grad_map, split=False)
                grad_map_lh, grad_map_rh = (
                    grad_map_full[:hemi_vertices],
                    grad_map_full[hemi_vertices:],
                )

                grad_img_lh = GiftiImage()
                grad_img_rh = GiftiImage()
                grad_img_lh.add_gifti_data_array(GiftiDataArray(grad_map_lh))
                grad_img_rh.add_gifti_data_array(GiftiDataArray(grad_map_rh))

                # Write cortical gradient to Gifti files
                nib.save(grad_img_lh, grad_map_lh_fn)
                nib.save(grad_img_rh, grad_map_rh_fn)

        grad_segments.append(grad_maps)

    grad_seg_dict[f"{method.lower()}_grad_segments"] = grad_segments

    return grad_seg_dict
