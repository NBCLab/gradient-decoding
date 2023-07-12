import argparse
import gc
import gzip
import itertools
import os
import os.path as op
import pickle

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from gradec.fetcher import _fetch_metamaps
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from neuromaps.datasets import fetch_fslr
from nilearn import datasets, image, masking, plotting
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from surfplot import Plot
from surfplot.utils import add_fslr_medial_wall, threshold
from wordcloud import WordCloud


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


def trim_image(img=None, tol=1, fix=True):
    mask = img != tol if fix else img <= tol
    if img.ndim == 3:
        mask = mask.any(2)
    mask0, mask1 = mask.any(0), mask.any(1)
    mask1[0] = False
    mask1[-1] = False
    return img[:, mask0]


def plot_top_words(topic_word_weight, features_name, n_top_words, dpi, out_filename):
    top_features_ind = topic_word_weight.argsort()[: -n_top_words - 1 : -1]
    top_features = [features_name[i] for i in top_features_ind]
    weights = topic_word_weight[top_features_ind]

    fig, ax = plt.subplots(figsize=(7, 9))

    norm = plt.Normalize(0, np.max(weights))
    color = plt.colormaps["YlOrRd"]
    colors = [color(norm(weight)) for weight in weights]

    ax.barh(top_features, weights, height=0.7, color=colors)
    ax.set_title("Topic-word weight", fontdict={"fontsize": 25})
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=20)

    fig.subplots_adjust(left=0.35, right=0.98, top=0.95, bottom=0.05)
    fig.savefig(out_filename, dpi=dpi)
    fig = None
    plt.close()
    gc.collect()
    plt.clf()


def plot_meta_maps(meta_maps_img, threshold, dpi, out_filename):
    template = datasets.load_mni152_template(resolution=1)

    display_modes = ["x", "y", "z"]
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    gs = GridSpec(2, 2, figure=fig)

    for dsp_i, display_mode in enumerate(display_modes):
        if display_mode == "z":
            ax = fig.add_subplot(gs[:, 1], aspect="equal")
        else:
            ax = fig.add_subplot(gs[dsp_i, 0], aspect="equal")

        plotting.plot_stat_map(
            meta_maps_img,
            bg_img=template,
            black_bg=False,
            draw_cross=False,
            annotate=True,
            threshold=threshold,
            colorbar=False,
            display_mode=display_mode,
            cut_coords=1,
            axes=ax,
        )
    fig.savefig(out_filename, bbox_inches="tight", dpi=dpi)
    fig = None
    plt.close()
    gc.collect()
    plt.clf()


def plot_surf_maps(lh_grad, rh_grad, threshold_, color_range, cmap, dpi, data_dir, out_filename):
    neuromaps_dir = op.join(data_dir, "neuromaps")

    surfaces = fetch_fslr(density="32k", data_dir=neuromaps_dir)
    lh, rh = surfaces["inflated"]
    sulc_lh, sulc_rh = surfaces["sulc"]

    lh_grad = threshold(lh_grad, threshold_)
    rh_grad = threshold(rh_grad, threshold_)

    p = Plot(surf_lh=lh, surf_rh=rh, layout="grid")
    p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)
    p.add_layer(
        {"left": lh_grad, "right": rh_grad},
        cmap=cmap,
        cbar=True,
        color_range=color_range,
    )
    fig = p.build()

    fig.savefig(out_filename, bbox_inches="tight", dpi=dpi)
    fig = None
    plt.close()
    gc.collect()
    plt.clf()


def word_cloud(frequencies_dict, dpi, data_dir, out_filename):
    class_dir = op.join(data_dir, "classification")
    classes_ = ["Functional", "Anatomical", "Clinical", "Non-Specific"]
    colors_ = ["#000000", "#0F6292", "#85CDFD", "#D3D3D3"]  # F7F7F7
    colors_dict = dict(zip(classes_, colors_))

    term_ns_df = pd.read_csv(
        op.join(class_dir, "term_neuroquery_classification.csv"),
        index_col="FEATURE",
    )
    term_nq_df = pd.read_csv(
        op.join(class_dir, "term_neuroquery_classification.csv"),
        index_col="FEATURE",
    )

    def color_func(word, font_size, position, orientation, font_path, random_state):
        if word in term_ns_df.index:
            class_ = term_ns_df.loc[[word], "Classification"].values[0]
        elif word in term_nq_df.index:
            class_ = term_nq_df.loc[[word], "Classification"].values[0]
        else:
            class_ = "Non-Specific"
        return colors_dict[class_]

    w = 6
    h = 5
    fig, ax = plt.subplots(figsize=(w, h))

    wc = WordCloud(
        width=w * dpi,
        height=h * dpi,
        background_color="white",
        random_state=0,
        colormap="YlOrRd",
    )
    wc.generate_from_frequencies(frequencies=frequencies_dict)
    wc.recolor(color_func=color_func)

    ax.imshow(wc)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    patches = [mpatches.Patch(color=color, label=label) for label, color in colors_dict.items()]
    fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=2, fontsize=12)

    fig.savefig(out_filename, bbox_inches="tight", dpi=dpi)
    fig = None
    plt.close()
    gc.collect()
    plt.clf()


def model_figures(
    data_dir,
    model,
    maps_fslr,
    class_df,
    output_dir,
    decoder=None,
    dset=None,
    cmap="afmhot",
    dpi=500,
    save_=False,
):
    full_vertices = 64984
    hemi_vertices = full_vertices // 2
    n_top_words = 10
    n_cols = 8
    n_rows = 1
    w = 8.5
    h = 2
    if decoder:
        meta_maps = decoder.images_

        features = [f.split("__")[-1] for f in decoder.features_]
        p_topic_g_word_df = model.distributions_["p_topic_g_word_df"]
        features_names = p_topic_g_word_df.columns.tolist()
        meta_maps_imgs = decoder.masker.inverse_transform(meta_maps)
        topic_word_weights = model.model.components_
        n_topics = len(topic_word_weights)

        threshold = 2  # threshold for the LDA models

    else:
        topic_word_weights = model.p_word_g_topic_.T
        n_topics = topic_word_weights.shape[0]
        vocabulary = np.array(model.vocabulary)
        sorted_weights_idxs = np.argsort(-topic_word_weights, axis=1)
        top_tokens = [
            "_".join(vocabulary[sorted_weights_idxs[topic_i, :]][:3])
            for topic_i in range(n_topics)
        ]

        features = [f"{i + 1}_{top_tokens[i]}" for i in range(n_topics)]
        features_names = model.vocabulary
        meta_maps_imgs = masking.unmask(model.p_topic_g_voxel_.T, model.mask)

    max_ = 15  # max numnber of characters in a feature name
    modified_list = [s if len(s) < max_ else s.replace(" ", "\n") for s in features_names]

    for topic_i in range(n_topics):
        feature = features[topic_i]
        feature_nm = feature.replace(" ", "-")
        print(f"\includegraphics[scale=1]{{{feature_nm}.eps}}\n")

        out_file = op.join(output_dir, f"{feature_nm}.eps")
        if not op.exists(out_file):
            if dset:
                n_docs = len(
                    dset.get_studies_by_label(labels=[f"LDA200__{feature}"], label_threshold=0.05)
                )
            else:
                threshold = np.percentile(maps_fslr[topic_i, :], 80)
                # print(f"Threshold: {threshold}")

            data = maps_fslr[topic_i, :]
            max_val = round(np.max(np.abs(data)), 2)
            range_ = (-max_val, max_val) if dset else (0, max_val)

            data = add_fslr_medial_wall(data)
            data_lh, data_rh = data[:hemi_vertices], data[hemi_vertices:full_vertices]

            meta_maps_img = image.index_img(meta_maps_imgs, topic_i)
            topic_word_weight = topic_word_weights[topic_i]

            frequencies_dict = dict(zip(features_names, topic_word_weight))

            temp_dir = op.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            out_top = op.join(temp_dir, "top.tiff")
            out_meta = op.join(temp_dir, "meta.tiff")
            out_surf = op.join(temp_dir, "surf.tiff")
            out_cloud = op.join(temp_dir, "cloud.tiff")

            plot_top_words(topic_word_weight, modified_list, n_top_words, dpi, out_top)
            plot_meta_maps(meta_maps_img, threshold, dpi, out_meta)
            plot_surf_maps(data_lh, data_rh, threshold, range_, cmap, dpi, data_dir, out_surf)
            word_cloud(frequencies_dict, dpi, data_dir, out_cloud)

            fig = plt.figure(figsize=(w, h))
            fig.subplots_adjust(
                left=None, bottom=None, right=None, top=None, wspace=None, hspace=None
            )
            gs = GridSpec(n_rows, n_cols, figure=fig)
            row = 0
            for img_file in [out_top, out_meta, out_surf, out_cloud]:
                img = mpimg.imread(img_file)

                ax = fig.add_subplot(gs[:, row : row + 2], aspect="equal")
                ax.imshow(img)
                ax.set_axis_off()

                row += 2

            # Conform title
            sub_features = feature.split("_")[1:]
            sub_features = [s.replace(" ", "_") for s in sub_features]
            title_lb = ", ".join(sub_features)
            docs_lb = f" (N={n_docs} docs)" if dset else ""
            class_ = class_df.loc[[feature], "Classification"].values[0]
            fig.suptitle(f'Topic {topic_i+1:03d}: "{title_lb}"{docs_lb}. {class_}.', fontsize=7)

            # Make sure the axis size if the same for different labels sizes
            plt.subplots_adjust(top=0.90)
            if save_:
                fig.savefig(out_file, bbox_inches="tight", dpi=dpi)
                plt.close()
            else:
                plt.show()
            plt.clf()
            gc.collect()


def load_files(dset_nm, model_nm, project_dir):
    data_dir = op.join(project_dir, "data")
    model_dir = op.join(data_dir, "models")
    meta_dir = op.join(data_dir, "meta-analysis")
    class_dir = op.join(data_dir, "classification")

    model_fn = op.join(model_dir, f"{model_nm}_{dset_nm}_model.pkl.gz")
    model_file = gzip.open(model_fn, "rb")
    model = pickle.load(model_file)

    if model_nm == "lda":
        decoder_fn = op.join(model_dir, f"lda_{dset_nm}_decoder.pkl.gz")
        decoder_file = gzip.open(decoder_fn, "rb")
        decoder = pickle.load(decoder_file)

        dset_fn = op.join(meta_dir, f"{dset_nm}_lda_dataset.pkl.gz")
        dset_file = gzip.open(dset_fn, "rb")
        dset = pickle.load(dset_file)
    else:
        decoder = None
        dset = None

    class_df = pd.read_csv(
        op.join(class_dir, f"{model_nm}_{dset_nm}_classification.csv"), index_col="FEATURE"
    )

    output_dir = op.join(project_dir, "figures", "Fig", "models", f"{model_nm}_{dset_nm}")
    os.makedirs(output_dir, exist_ok=True)

    maps_fslr = _fetch_metamaps(dset_nm, model_nm, data_dir=data_dir)

    return model, decoder, dset, class_df, maps_fslr, output_dir


def main(project_dir, n_cores):
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)

    DPI = 500

    # Define Paths
    # =============
    project_dir = op.abspath("/Users/jperaza/Documents/GitHub/gradient-decoding")
    data_dir = op.join(project_dir, "data")

    # dset_nms = ["neurosynth", "neuroquery"]
    # model_nms = ["lda", "gclda"]
    dset_nms = ["neuroquery"]
    model_nms = ["gclda"]
    for dset_nm, model_nm in itertools.product(dset_nms, model_nms):
        cmap = nilearn_cmaps["cold_hot"] if model_nm == "lda" else "afmhot"
        # threshold = 2 if model_nm == "lda" else 0.001

        print(dset_nm, flush=True)
        model, decoder, dset, classification, maps_fslr, output_dir = load_files(
            dset_nm,
            model_nm,
            project_dir,
        )

        model_figures(
            data_dir,
            model,
            maps_fslr,
            classification,
            output_dir,
            decoder=decoder,
            dset=dset,
            cmap=cmap,
            dpi=DPI,
            save_=True,
        )


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
