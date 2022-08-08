"""Mata-analytic decoding tools."""
import logging
import os.path as op
from glob import glob

import pandas as pd
from nimare.annotate.lda import LDAModel
from nimare.utils import get_resource_path
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

LGR = logging.getLogger(__name__)


def _generate_counts(
    text_df, vocabulary=None, text_column="abstract", tfidf=True, min_df=0.01, max_df=0.99
):
    """Generate tf-idf weights for unigrams/bigrams derived from textual data.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        A DataFrame with two columns ('id' and 'text'). D = document.

    Returns
    -------
    weights_df : (D x T) :obj:`pandas.DataFrame`
        A DataFrame where the index is 'id' and the columns are the
        unigrams/bigrams derived from the data. D = document. T = term.
    """
    if text_column not in text_df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    # Remove rows with empty text cells
    orig_ids = text_df["id"].tolist()
    text_df = text_df.fillna("")
    keep_ids = text_df.loc[text_df[text_column] != "", "id"]
    text_df = text_df.loc[text_df["id"].isin(keep_ids)]

    if len(keep_ids) != len(orig_ids):
        LGR.info(f"Retaining {len(keep_ids)}/{len(orig_ids)} studies")

    ids = text_df["id"].tolist()
    text = text_df[text_column].tolist()
    stoplist = op.join(get_resource_path(), "neurosynth_stoplist.txt")
    with open(stoplist, "r") as fo:
        stop_words = fo.read().splitlines()

    if tfidf:
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            vocabulary=vocabulary,
            stop_words=stop_words,
        )
    else:
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            vocabulary=vocabulary,
            stop_words=stop_words,
        )
    weights = vectorizer.fit_transform(text).toarray()

    if hasattr(vectorizer, "get_feature_names_out"):
        # scikit-learn >= 1.0.0
        names = vectorizer.get_feature_names_out()
    else:
        # scikit-learn < 1.0.0
        # To remove when we drop support for 3.6 and increase minimum sklearn version to 1.0.0.
        names = vectorizer.get_feature_names()

    names = [str(name) for name in names]
    weights_df = pd.DataFrame(weights, columns=names, index=ids)
    weights_df.index.name = "id"
    return weights_df


def annotate_lda(dset, dataset, data_dir, n_topics=200, n_cores=1):
    """Generate tf-idf weights for unigrams/bigrams derived from textual data.

    Parameters
    ----------
    dset : :obj:`~nimare.dataset.Dataset`
        A Dataset with, at minimum, text available in the ``self.text_column`` column of its
        :py:attr:`~nimare.dataset.Dataset.texts` attribute.
    n_topics : :obj:`int`
        Number of topics for topic model. This corresponds to the model's ``n_components``
        parameter. Must be an integer >= 1.
    dataset: str
        Dataset name. Possible options: "neurosynth" or "neuroquery"
    data_dir: str
        Path to data directory.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Returns
        -------
        dset : :obj:`~nimare.dataset.Dataset`
            A new Dataset with an updated :py:attr:`~nimare.dataset.Dataset.annotations` attribute.
    """
    if dataset == "neurosynth":
        feature_group = "terms_abstract_tfidf"
        feature_names = dset.annotations.columns.values
        feature_names = [f for f in feature_names if f.startswith(feature_group)]
        vocabulary = [f.split("__")[-1] for f in feature_names]
        counts_df = _generate_counts(
            dset.texts,
            vocabulary=vocabulary,
            text_column="abstract",
            tfidf=False,
            max_df=len(dset.ids) - 2,
            min_df=2,
        )

    elif dataset == "neuroquery":
        counts_dir = op.join(data_dir, "neuroquery", "neuroquery_counts")
        counts_arr_fns = glob(op.join(counts_dir, "*_features.npz"))
        counts_sparse = None
        for file_i, counts_arr_fn in enumerate(counts_arr_fns):
            if file_i == 0:
                counts_sparse = load_npz(counts_arr_fn)
            else:
                counts_sparse = counts_sparse + load_npz(counts_arr_fn)
        counts_arr = counts_sparse.todense()

        ids = dset.annotations["id"].tolist()
        feature_names = dset.inputs_["annotations"].columns.values
        counts_df = pd.DataFrame(counts_arr, columns=feature_names, index=ids)
        counts_df.index.name = "id"

    model = LDAModel(n_topics=n_topics, max_iter=1000, n_cores=n_cores)
    new_dset = model.fit(dset, counts_df)

    return new_dset
