import gzip
import os.path as op
import pickle
from difflib import get_close_matches
from glob import glob

import numpy as np
import pandas as pd
from gradec.fetcher import _fetch_features
from sklearn.feature_extraction.text import TfidfTransformer


def _extract_semantic_similarity(max_feature, ic_df, tfidf_df, frequency_threshold):
    include_rows = tfidf_df[max_feature] >= frequency_threshold
    include_tfidf = tfidf_df[max_feature][include_rows]

    return ic_df[max_feature].values[0], include_tfidf.mean(axis=0)


def _get_semantic_similarity(
    model_nm, ic_df, tfidf_df, max_features, frequency_threshold, n_top_terms
):
    ic_lst, tfidf_lst = [], []
    for max_feature in max_features:
        if model_nm == "term":
            ic, tfidf = _extract_semantic_similarity(
                max_feature, ic_df, tfidf_df, frequency_threshold
            )

        else:
            sub_max_features = max_feature.split("_")  # [1:] when the index is included
            assert len(sub_max_features) == n_top_terms

            sub_ic_lst, sub_tfidf_lst = [], []
            for sub_max_feature in sub_max_features:
                sub_ic, sub_tfidf = _extract_semantic_similarity(
                    sub_max_feature, ic_df, tfidf_df, frequency_threshold
                )
                sub_ic_lst.append(sub_ic)
                sub_tfidf_lst.append(sub_tfidf)

            ic = np.sum(sub_ic_lst)
            tfidf = np.sum(sub_tfidf_lst)

        ic_lst.append(ic)
        tfidf_lst.append(tfidf)

    return ic_lst, tfidf_lst


def _find_category(classification_df, term, idx_col, col_name):
    if classification_df.index.name != idx_col:
        classification_df = classification_df.set_index(idx_col)
    classification_df.index = classification_df.index.astype(str)

    m = classification_df.index == term
    if m.sum() > 0:
        classification = classification_df.loc[term, col_name]
    else:
        val = get_close_matches(term, classification_df.index)
        if len(val) <= 0:
            return None
        classification = classification_df.loc[val[0], col_name]
    if isinstance(classification, pd.Series):
        classification = classification.to_list()[0]

    return classification


def neuroquery_annot(features, ns_classification_df, nq_classification_df):
    nq_2_ns = {
        "anatomy": "Anatomical",
        "disease": "Clinical",
        "psychology": "Functional",
    }

    classification_lst = []
    for term in features:
        classification = _find_category(
            ns_classification_df, term, "FEATURE", "Classification"
        )
        if classification is None:
            for idx_col in ["term", "normalized_term"]:
                classification = _find_category(
                    nq_classification_df, term, idx_col, "category"
                )
                if classification is not None:
                    break

            if classification is not None:
                classification = nq_2_ns[classification]

        if classification is None:
            classification = "Non-Specific"

        classification_lst.append(classification)

    classification_df = pd.DataFrame()
    classification_df["FEATURE"] = features
    classification_df["Classification"] = classification_lst

    classification_df = classification_df.set_index("FEATURE")

    return classification_df


def _nq_term_classifier(data_dir, features, ns_term_class_df, nq_term_class_fn):
    nq_categories_fn = op.join(
        data_dir, "raw", "data-neuroquery_version-1_termcategories.csv"
    )
    nq_categories_df = pd.read_csv(nq_categories_fn)

    result = neuroquery_annot(features, ns_term_class_df, nq_categories_df)
    result.to_csv(nq_term_class_fn)
    return result


def crowdsourced_annot(crowdsourced_files):
    # Majority voting algorithm
    for i, files in enumerate(crowdsourced_files):
        if i == 0:
            pd_concat = pd.read_csv(files, index_col="FEATURE").fillna(0)
            pd_concat = pd_concat.replace(["X", "x"], 1)
        else:
            ind_classification = pd.read_csv(files, index_col="FEATURE").fillna(0)
            ind_classification = ind_classification.replace(["X", "x"], 1)
            pd_concat = pd.concat([pd_concat, ind_classification], axis=0)

    # TODO: error when pd_concat.dtypes != int64
    terms_classified_df = pd_concat.dropna(axis=1).groupby("FEATURE").mean()
    terms_classified_df["Classification"] = terms_classified_df.idxmax(axis=1)

    return terms_classified_df


def term_classifier(terms, terms_classified_df):
    classification = []
    for term in terms:
        if term in terms_classified_df.index:
            row = terms_classified_df.loc[[term]]
            classification.append(row["Classification"].values[0])
        else:
            classification.append("Non-Specific")

    return np.array(classification)


def topic_classifier(terms, n_top_terms, weights, terms_classified_df):
    cotegories = np.array(["Functional", "Clinical", "Anatomical", "Non-Specific"])
    classification_lst = []
    for term in terms:
        sub_max_features = term.split("_")[1:]
        assert len(sub_max_features) == n_top_terms

        cotegories_count = np.zeros(len(cotegories))
        for sub_max_feature, weight in zip(sub_max_features, weights):
            if sub_max_feature in terms_classified_df.index:
                row = terms_classified_df.loc[[sub_max_feature]]
                sub_classification = row["Classification"].values[0]
            else:
                sub_classification = "Non-Specific"
            sub_class_idx = np.where(cotegories == sub_classification)[0]
            cotegories_count[sub_class_idx] += 1 * weight
        class_sorted = np.argsort(-cotegories_count)

        classification = cotegories[class_sorted][0]
        classification_lst.append(classification)

    classification_df = pd.DataFrame()
    classification_df["FEATURE"] = terms
    classification_df["Classification"] = classification_lst

    classification_df = classification_df.set_index("FEATURE")

    return np.array(classification_lst), classification_df


def classifier(terms, n_top_terms, weights, dset_nm, model_nm, data_dir):
    class_dir = op.join(data_dir, "classification")
    ns_term_class_fn = op.join(class_dir, "term_neurosynth_classification.csv")

    if not op.isfile(ns_term_class_fn):
        crowdsourced_files = sorted(
            glob(
                op.join(
                    class_dir, "raw", "CrowdsourcedNeurosynthTermClassifications-*.*"
                )
            )
        )
        ns_term_class_df = crowdsourced_annot(crowdsourced_files)
        ns_term_class_df.to_csv(ns_term_class_fn)
    else:
        ns_term_class_df = pd.read_csv(ns_term_class_fn, index_col="FEATURE")

    if dset_nm == "neurosynth":
        term_class_df = ns_term_class_df.copy()

    elif dset_nm == "neuroquery":
        nq_term_features = _fetch_features("neuroquery", "term", data_dir=data_dir)
        nq_term_class_fn = op.join(class_dir, "term_neuroquery_classification.csv")
        nq_term_class_df = (
            pd.read_csv(nq_term_class_fn, index_col="FEATURE")
            if op.isfile(nq_term_class_fn)
            else _nq_term_classifier(
                class_dir, nq_term_features, ns_term_class_df, nq_term_class_fn
            )
        )
        term_class_df = nq_term_class_df.copy()

    if model_nm == "term":
        term_classified = term_classifier(terms, term_class_df)
    else:
        topic_class_fn = op.join(class_dir, f"{model_nm}_{dset_nm}_classification.csv")
        if not op.isfile(topic_class_fn):
            term_classified, topic_class_df = topic_classifier(
                terms, n_top_terms, weights, term_class_df
            )
            topic_class_df.to_csv(topic_class_fn)
        else:
            topic_class_df = pd.read_csv(topic_class_fn, index_col="FEATURE")
            term_classified = term_classifier(terms, topic_class_df)

    return term_classified


def _get_ic(counts_df):
    p_t_c = counts_df.sum(axis=0) / counts_df.values.sum()
    ic_df = -np.log(p_t_c)
    ic_df = ic_df.replace([np.inf, -np.inf], 0)
    ic_df = ic_df.to_frame().T

    return ic_df


def _get_tfidf(counts_df):
    tfidf_tr = TfidfTransformer()
    X = counts_df.to_numpy()
    X_tr = tfidf_tr.fit_transform(X)
    return pd.DataFrame(
        X_tr.toarray(), index=counts_df.index, columns=counts_df.columns
    )


def _combine_counts(class_data_dir):
    ns_counts_df_fn = op.join(class_data_dir, "neurosynth_counts.tsv")
    nq_counts_df_fn = op.join(class_data_dir, "neuroquery_counts.tsv")
    ns_counts_df = pd.read_csv(ns_counts_df_fn, delimiter="\t", index_col="id")
    nq_counts_df = pd.read_csv(nq_counts_df_fn, delimiter="\t", index_col="id")

    counts_df = pd.merge(nq_counts_df, ns_counts_df, how="outer", on=["id"])
    counts_df = counts_df.fillna(0)
    for col in counts_df.columns:
        if (col.endswith("_x") or col.endswith("_y")) and col in counts_df.columns:
            counts_df[col[:-2]] = (
                counts_df[col[:-2] + "_x"] + counts_df[col[:-2] + "_y"]
            )
            counts_df.drop([col[:-2] + "_x", col[:-2] + "_y"], axis=1, inplace=True)
    counts_df = counts_df.sort_index(axis=1)
    return counts_df


def _get_twfrequencies(dset_nm, model_nm, n_top_terms, dec_data_dir):
    model_fn = op.join(dec_data_dir, f"{model_nm}_{dset_nm}_model.pkl.gz")
    model_file = gzip.open(model_fn, "rb")
    model_obj = pickle.load(model_file)

    topic_word_weights = (
        model_obj.p_word_g_topic_.T
        if model_nm == "gclda"
        else model_obj.distributions_["p_topic_g_word"]
    )

    n_topics = topic_word_weights.shape[0]
    sorted_weights_idxs = np.argsort(-topic_word_weights, axis=1)
    frequencies_lst = []
    for topic_i in range(n_topics):
        frequencies = topic_word_weights[topic_i, sorted_weights_idxs[topic_i, :]][
            :n_top_terms
        ].tolist()
        frequencies = [freq / np.max(frequencies) for freq in frequencies]
        frequencies = np.round(frequencies, 3).tolist()
        frequencies_lst.append(frequencies)

    return frequencies_lst
