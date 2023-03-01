import os.path as op
from difflib import get_close_matches
from glob import glob

import numpy as np
import pandas as pd


def _get_semantic_similarity(ic_df, tfidf_df, max_feature, max_feature_lb, frequency_threshold):
    include_rows = tfidf_df[max_feature_lb] >= frequency_threshold
    include_ic = ic_df[max_feature][include_rows]
    include_tfidf = tfidf_df[max_feature_lb][include_rows]

    ic = include_ic.mean(axis=0)
    tfidf = include_tfidf.mean(axis=0)

    return ic, tfidf


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
    nq_2_ns = {"anatomy": "Anatomical", "disease": "Clinical", "psychology": "Functional"}

    classification_lst = []
    for term in features:
        classification = _find_category(ns_classification_df, term, "FEATURE", "Classification")
        if classification is None:
            for idx_col in ["term", "normalized_term"]:
                classification = _find_category(nq_classification_df, term, idx_col, "category")
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


def topic_classifier(terms, terms_classified_df):
    cotegories = np.array(["Functional", "Clinical", "Anatomical", "Non-Specific"])
    classification_lst = []
    for term in terms:
        sub_max_features = term.split("_")[1:]
        assert len(sub_max_features) == 3

        cotegories_count = np.zeros(len(cotegories))
        for sub_max_feature in sub_max_features:
            if sub_max_feature in terms_classified_df.index:
                row = terms_classified_df.loc[[sub_max_feature]]
                sub_classification = row["Classification"].values[0]
            else:
                sub_classification = "Non-Specific"
            sub_class_idx = np.where(cotegories == sub_classification)[0]
            cotegories_count[sub_class_idx] += 1
        class_sorted = np.argsort(-cotegories_count)

        classification = cotegories[class_sorted][0]
        classification_lst.append(classification)

    classification_df = pd.DataFrame()
    classification_df["FEATURE"] = terms
    classification_df["Classification"] = classification_lst

    classification_df = classification_df.set_index("FEATURE")

    return np.array(classification_lst), classification_df


def classifier(terms, dset, model, dset_name, data_dir):
    ns_term_class_fn = op.join(data_dir, "term_neurosynth_classification.csv")

    if not op.isfile(ns_term_class_fn):
        crowdsourced_files = sorted(
            glob(op.join(data_dir, "raw", "CrowdsourcedNeurosynthTermClassifications-*.*"))
        )
        ns_term_class_df = crowdsourced_annot(crowdsourced_files)
        ns_term_class_df.to_csv(ns_term_class_fn)
    else:
        ns_term_class_df = pd.read_csv(ns_term_class_fn, index_col="FEATURE")

    if dset_name == "neurosynth":
        term_class_df = ns_term_class_df.copy()

    elif dset_name == "neuroquery":
        nq_term_class_fn = op.join(data_dir, "term_neuroquery_classification.csv")
        nq_term_class_df = (
            pd.read_csv(nq_term_class_fn, index_col="FEATURE")
            if op.isfile(nq_term_class_fn)
            else _extracted_from_classifier_19(
                data_dir, dset, ns_term_class_df, nq_term_class_fn
            )
        )
        term_class_df = nq_term_class_df.copy()

    if model == "term":
        term_classified = term_classifier(terms, term_class_df)
    else:
        topic_class_fn = op.join(data_dir, f"{model}_{dset_name}_classification.csv")
        if not op.isfile(topic_class_fn):
            term_classified, topic_class_df = topic_classifier(terms, term_class_df)
            topic_class_df.to_csv(topic_class_fn)
        else:
            topic_class_df = pd.read_csv(topic_class_fn, index_col="FEATURE")
            term_classified = term_classifier(terms, topic_class_df)

    return term_classified


# TODO Rename this here and in `classifier`
def _extracted_from_classifier_19(data_dir, dset, ns_term_class_df, nq_term_class_fn):
    nq_categories_fn = op.join(
        data_dir, "raw", "data-neuroquery_version-1_termcategories.csv"
    )
    nq_categories_df = pd.read_csv(nq_categories_fn)

    feature_names = dset.annotations.columns.values
    feature_names = [
        f for f in feature_names if f.startswith("neuroquery6308_combined_tfidf")
    ]
    features = [f.split("__")[-1] for f in feature_names]

    result = neuroquery_annot(features, ns_term_class_df, nq_categories_df)
    result.to_csv(nq_term_class_fn)
    return result
