"""Miscellaneous functions used for plotting gradient data"""
import gzip
import math
import os.path as op
import pickle

import nibabel as nib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from neuromaps.datasets import fetch_fslr
from nilearn import image, masking, plotting
from nilearn.plotting import plot_stat_map
from surfplot import Plot
from surfplot.utils import threshold


def plot_gradient(
    data_dir,
    grad_seg_fnames,
    grad_seg_labels=None,
    cmap="viridis",
    threshold_=None,
    color_range=None,
    views=None,
    title=False,
    layout="grid",
    cbar=False,
    out_dir=None,
    prefix="",
):
    prefix_sep = "" if prefix == "" else "_"
    if not prefix.endswith(prefix_sep):
        prefix = prefix + prefix_sep

    neuromaps_dir = op.join(data_dir, "neuromaps-data")
    surfaces = fetch_fslr(density="32k", data_dir=neuromaps_dir)

    lh, rh = surfaces["inflated"]
    sulc_lh, sulc_rh = surfaces["sulc"]

    for img_i, (grad_segment_lh, grad_segment_rh) in enumerate(grad_seg_fnames):
        lh_grad = nib.load(grad_segment_lh).agg_data()
        rh_grad = nib.load(grad_segment_rh).agg_data()

        if threshold_ is not None:
            lh_grad = threshold(lh_grad, threshold_)
            rh_grad = threshold(rh_grad, threshold_)

        if views:
            p = Plot(surf_lh=lh, views=views, layout=layout)
            p.add_layer({"left": sulc_lh}, cmap="binary_r", cbar=False)
            p.add_layer({"left": lh_grad}, cmap=cmap, cbar=cbar, color_range=color_range)
        else:
            p = Plot(surf_lh=lh, surf_rh=rh, layout=layout)
            p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)
            p.add_layer(
                {"left": lh_grad, "right": rh_grad}, cmap=cmap, cbar=cbar, color_range=color_range
            )

        fig = p.build()
        if grad_seg_labels is None:
            base_name = op.basename(grad_segment_lh)
            firts_name = base_name.split("_")[0].split("-")[1]
            last_name = base_name.split("_")[1].split("-")[1]
            id_name = base_name.split("_")[2].split("-")[1]
            title_ = f"{firts_name}-{last_name}"
        else:
            title_ = grad_seg_labels[img_i]

        if title:
            fig.axes[0].set_title(title_, pad=-3)

        if out_dir is not None:
            out_file = op.join(out_dir, f"{prefix}{title_}.tiff")
            plt.savefig(out_file, bbox_inches="tight", dpi=1000)

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


def plot_meta_maps(
    decoder_fn, map_idxs, threshold=2, model="decoder", colorbar=True, out_dir=None
):
    decoder_file = gzip.open(decoder_fn, "rb")
    decoder = pickle.load(decoder_file)
    if model == "decoder":
        meta_maps = decoder.images_
        features = [f.split("__")[-1] for f in decoder.features_]
        meta_maps_imgs = decoder.masker.inverse_transform(meta_maps[map_idxs, :])
    elif model == "gclda":
        topic_word_weights = decoder.p_word_g_topic_
        n_topics = topic_word_weights.shape[1]
        vocabulary = np.array(decoder.vocabulary)
        sorted_weights_idxs = np.argsort(-topic_word_weights, axis=0)
        top_tokens = [
            "_".join(vocabulary[sorted_weights_idxs[:, topic_i]][:3])
            for topic_i in range(n_topics)
        ]
        features = [f"{i + 1}_{top_tokens[i]}" for i in range(n_topics)]
        meta_maps_imgs = masking.unmask(decoder.p_voxel_g_topic_.T[map_idxs, :], decoder.mask)

    features_to_plot = np.array(features)[map_idxs]
    n_maps = len(map_idxs)
    for i_feature in range(n_maps):
        feature_img_3d = image.index_img(meta_maps_imgs, i_feature)
        plotting.plot_stat_map(
            feature_img_3d,
            draw_cross=False,
            colorbar=colorbar,
            annotate=False,
            threshold=threshold,
            title=features_to_plot[i_feature],
        )
        if out_dir is not None:
            out_file = op.join(out_dir, f"{features_to_plot[i_feature]}.tiff")
            plt.savefig(out_file, bbox_inches="tight", dpi=1000)

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


def plot_profile(data_df, metric, hue_order, cmap="tab20"):
    # sns.set(style="whitegrid")

    my_cmap = plt.get_cmap(cmap)
    n_segments = 30
    fig, axes = plt.subplots(n_segments, 2)
    fig.set_size_inches(15, 90)

    for seg_sol in range(n_segments):
        test_df = data_df[data_df["segment_solution"] == seg_sol + 3]
        sub_data_df = test_df.drop(columns=["segment_solution", "pvalue"])
        sub_data_df = sub_data_df.pivot_table(
            values=metric, index=sub_data_df["segment"], columns="method"
        )
        sub_data_df = sub_data_df.reindex(hue_order, axis=1)

        sub_data_df.plot.bar(
            rot=0,
            width=1,
            stacked=True,
            color=my_cmap.colors[: len(hue_order)],
            ax=axes[seg_sol, 0],
        )
        axes[seg_sol, 0].get_legend().remove()

        if seg_sol == 0:
            handles, labels = axes[0, 0].get_legend_handles_labels()

        test_df = test_df.reset_index()
        test_df["segment"] = test_df["segment"].astype(str)
        """
        sns.barplot(
            data=test_df, 
            x="segment", 
            y="max_corr", 
            palette=cmap, 
            hue="method", 
            hue_order=hue_order, 
            dodge=True,
            ax=axes[seg_sol , 1],
        )
        axes[seg_sol , 1].get_legend().remove()
        """
        # x = test_df["segment"]
        # y = test_df["max_corr"]
        # axes[seg_sol , 2].plot(x, y, 'o-')
        # axes[seg_sol , 2].get_legend().remove()
        sns.lineplot(
            data=test_df,
            x="segment",
            y=metric,
            palette=cmap,
            hue="method",
            hue_order=hue_order,
            marker="o",
            ax=axes[seg_sol, 1],
        )
        axes[seg_sol, 1].get_legend().remove()

        text_lst = []
        mean_lst = []
        for approach in hue_order:
            approach_df = test_df[test_df["method"] == approach]
            # print(approach_df)
            mean_corr = approach_df[metric]
            text_lst.append(f"{mean_corr.mean():.3f} ± {mean_corr.std():.3f}")
            mean_lst.append(mean_corr.mean())

        ax_handles, ax_labels = axes[seg_sol, 1].get_legend_handles_labels()
        sort_idx = np.argsort(-np.array(mean_lst))
        """
        axes[seg_sol, 1].legend(
            np.array(ax_handles)[sort_idx],
            np.array(text_lst)[sort_idx],
            loc="upper left",
            bbox_to_anchor=(1.04, 1.15),
            ncol=1,
        )
        """
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout()
    # plt.savefig(op.join(result_dir, "gradient_segmentation", "Figures", "correlation_profile.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_mean_profile(data_df, metric, hue_order, cmap="tab20"):
    # sns.set(style="whitegrid")

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3, 15)

    sns.lineplot(
        data=data_df,
        x=metric,
        y="segment_solution",
        palette=cmap,
        hue="method",
        hue_order=hue_order,
        sort=False,
        marker="o",
        ax=ax,
    )
    ax.get_legend().remove()
    # plt.savefig(op.join("./Fig", "mean_correlation_profile.eps"), bbox_inches="tight")
    plt.show()
