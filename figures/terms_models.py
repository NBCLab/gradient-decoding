import argparse
import gc
import gzip
import os
import os.path as op
import pickle
from glob import glob

import matplotlib
import matplotlib.image as mpimg
import nibabel as nib
import numpy as np
import pandas as pd
from gradec.fetcher import _fetch_metamaps

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from surfplot.utils import add_fslr_medial_wall
from topics_models import plot_meta_maps, plot_surf_maps


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


def model_figures(
    data_dir,
    dset,
    dset_nm,
    maps_fslr,
    class_df,
    output_dir,
    meta_maps_fns,
    cmap="afmhot",
    threshold=2,
    dpi=500,
    save_=False,
):
    full_vertices = 64984
    hemi_vertices = full_vertices // 2
    n_cols = 2
    n_rows = 1
    w = 4.5
    h = 2

    # meta_maps = decoder.images_

    features = [op.basename(f).split(".")[0] for f in meta_maps_fns]
    # features = [f.split("__")[-1] for f in decoder.features_]
    # meta_maps_imgs = decoder.masker.inverse_transform(meta_maps)
    class_dir = op.join(data_dir, "classification")
    term_ns_df = pd.read_csv(
        op.join(class_dir, "term_neuroquery_classification.csv"),
        index_col="FEATURE",
    )
    term_nq_df = pd.read_csv(
        op.join(class_dir, "term_neuroquery_classification.csv"),
        index_col="FEATURE",
    )

    n_maps = len(meta_maps_fns)
    for map_i in range(n_maps):
        feature = features[map_i]
        feature_nm = feature.replace(" ", "-")
        print(f"\includegraphics[scale=1]{{{map_i:04d}_{feature_nm}.eps}}")
        if map_i % 2 != 0:
            print("")

        out_file = op.join(output_dir, f"{map_i:04d}_{feature_nm}.eps")
        if not op.exists(out_file):
            if dset_nm == "neurosynth":
                feature_group = "terms_abstract_tfidf"
            elif dset_nm == "neuroquery":
                feature_group = "neuroquery6308_combined_tfidf"

            frequency_threshold = 0.001
            n_docs = len(
                dset.get_studies_by_label(
                    labels=[f"{feature_group}__{feature}"], label_threshold=frequency_threshold
                )
            )

            data = maps_fslr[map_i, :]
            max_val = round(np.max(np.abs(data)), 2)
            range_ = (-max_val, max_val)

            data = add_fslr_medial_wall(data)
            data_lh, data_rh = data[:hemi_vertices], data[hemi_vertices:full_vertices]

            # meta_maps_img = image.index_img(meta_maps_imgs, map_i)
            meta_maps_img = nib.load(meta_maps_fns[map_i])

            temp_dir = op.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            out_meta = op.join(temp_dir, "meta.tiff")
            out_surf = op.join(temp_dir, "surf.tiff")

            plot_meta_maps(meta_maps_img, threshold, dpi, out_meta)
            plot_surf_maps(data_lh, data_rh, threshold, range_, cmap, dpi, data_dir, out_surf)

            fig = plt.figure(figsize=(w, h))
            fig.subplots_adjust(
                left=None, bottom=None, right=None, top=None, wspace=None, hspace=None
            )
            gs = GridSpec(n_rows, n_cols, figure=fig)
            for img_i, img_file in enumerate([out_meta, out_surf]):
                img = mpimg.imread(img_file)

                ax = fig.add_subplot(gs[0, img_i], aspect="equal")
                ax.imshow(img)
                ax.set_axis_off()

            # Conform title
            title_lb = feature.replace(" ", "_")
            docs_lb = f" (N={n_docs} docs)" if dset else ""

            if feature in term_ns_df.index:
                class_ = term_ns_df.loc[[feature], "Classification"].values[0]
            elif feature in term_nq_df.index:
                class_ = term_nq_df.loc[[feature], "Classification"].values[0]
            else:
                class_ = "Non-Specific"
            fig.suptitle(f'Term: "{title_lb}"{docs_lb}. {class_}.', fontsize=7)

            # Make sure the axis size if the same for different labels sizes
            plt.subplots_adjust(top=0.90)
            if save_:
                fig.savefig(out_file, bbox_inches="tight", dpi=dpi)
                plt.close()
            else:
                plt.show()
            plt.clf()
            gc.collect()


def load_files(dset_nm, project_dir):
    data_dir = op.join(project_dir, "data")
    meta_dir = op.join(data_dir, "meta-analysis")
    # model_dir = op.join(data_dir, "models")
    class_dir = op.join(data_dir, "classification")

    dset_fn = op.join(meta_dir, f"{dset_nm}_dataset.pkl.gz")
    dset_file = gzip.open(dset_fn, "rb")
    dset = pickle.load(dset_file)

    # decoder_fn = op.join(model_dir, f"term_{dset_nm}_decoder.pkl.gz")
    # decoder_file = gzip.open(decoder_fn, "rb")
    # decoder = pickle.load(decoder_file)
    meta_maps_fns = sorted(glob(op.join("/Users/jperaza/Documents/Data", dset_nm, "*.nii.gz")))

    class_df = pd.read_csv(
        op.join(class_dir, f"term_{dset_nm}_classification.csv"), index_col="FEATURE"
    )

    output_dir = op.join(project_dir, "figures", "Fig", "models", f"term_{dset_nm}")
    os.makedirs(output_dir, exist_ok=True)

    maps_fslr = _fetch_metamaps(dset_nm, "term", data_dir=data_dir)

    return dset, meta_maps_fns, class_df, maps_fslr, output_dir


def main(project_dir, n_cores):
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)

    DPI = 500

    # Define Paths
    # =============
    project_dir = op.abspath("/Users/jperaza/Documents/GitHub/gradient-decoding")
    data_dir = op.join(project_dir, "data")

    cmap = nilearn_cmaps["cold_hot"]
    threshold = 2

    # dset_nms = ["neurosynth", "neuroquery"]
    dset_nms = ["neuroquery"]
    for dset_nm in dset_nms:
        print(dset_nm, flush=True)
        dset, meta_maps_fns, class_df, maps_fslr, output_dir = load_files(dset_nm, project_dir)

        model_figures(
            data_dir,
            dset,
            dset_nm,
            maps_fslr,
            class_df,
            output_dir,
            meta_maps_fns,
            cmap=cmap,
            threshold=threshold,
            dpi=DPI,
            save_=True,
        )


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
