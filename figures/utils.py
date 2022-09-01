"""Miscellaneous functions used for plotting gradient data"""
import gzip
import math
import os.path as op
import pickle

import nibabel as nib
from matplotlib import pyplot as plt
from neuromaps.datasets import fetch_fslr
from nilearn import image, plotting
from nilearn.plotting import plot_stat_map
from surfplot import Plot
from surfplot.utils import threshold


def plot_gradient(data_dir, grad_seg_fnames, cmap="viridis", threshold_=None, color_range=None):
    neuromaps_dir = op.join(data_dir, "neuromaps-data")
    surfaces = fetch_fslr(density="32k", data_dir=neuromaps_dir)

    lh, rh = surfaces["inflated"]
    sulc_lh, sulc_rh = surfaces["sulc"]

    for grad_segment_lh, grad_segment_rh in grad_seg_fnames:
        lh_grad = nib.load(grad_segment_lh).agg_data()
        rh_grad = nib.load(grad_segment_rh).agg_data()

        if threshold_ is not None:
            lh_grad = threshold(lh_grad, threshold_)
            rh_grad = threshold(rh_grad, threshold_)

        p = Plot(surf_lh=lh, surf_rh=rh)
        p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)
        p.add_layer({"left": lh_grad, "right": rh_grad}, cmap=cmap, color_range=color_range)

        fig = p.build()
        base_name = op.basename(grad_segment_lh)
        firts_name = base_name.split("_")[0].split("-")[1]
        last_name = base_name.split("_")[1].split("-")[1]
        title_ = f"{firts_name}: {last_name}"
        fig.axes[0].set_title(title_, pad=-3)
        plt.show()


def plot_subcortical_gradient(subcort_grad_fnames, cmap="viridis", threshold_=None):
    for subcort_grad_fname in subcort_grad_fnames:
        base_name = op.basename(subcort_grad_fname)
        firts_name = base_name.split("_")[0].split("-")[1]
        last_name = base_name.split("_")[1].split("-")[1]
        title_ = f"{firts_name}: {last_name}"

        plot_stat_map(
            subcort_grad_fname,
            draw_cross=False,
            cmap=cmap,
            threshold=threshold_,
            title=title_,
        )
        plt.show()


def plot_meta_maps(term_based_decoder_fn, n_init=0, n_maps=10, threshold=2):
    term_decoder_file = gzip.open(term_based_decoder_fn, "rb")
    term_decoder = pickle.load(term_decoder_file)
    term_based_meta_maps = term_decoder.images_
    term_features = [f.split("__")[-1] for f in term_decoder.features_]

    term_meta_maps_imgs = term_decoder.masker.inverse_transform(
        term_based_meta_maps[n_init : n_init + n_maps, :]
    )
    term_features_to_plot = term_features[n_init : n_init + n_maps]

    for i_feature in range(n_maps):
        feature_img_3d = image.index_img(term_meta_maps_imgs, i_feature)
        plotting.plot_stat_map(
            feature_img_3d,
            draw_cross=False,
            colorbar=True,
            annotate=False,
            threshold=threshold,
            title=term_features_to_plot[i_feature],
        )
        plt.show()


def plot_top_words(model, feature_names, n_top_words, title):
    n_topics = len(model.components_)
    n_cols = 5
    n_rows = math.ceil(n_topics / n_cols)
    w = 30
    h = (w / 2) * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w, h), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.977, bottom=0.05, wspace=0.90, hspace=0.1)
    plt.show()
