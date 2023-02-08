import os.path as op
from difflib import get_close_matches
from glob import glob

import numpy as np
import pandas as pd


def _find_category(classification_df, term, idx_col, col_name):
    classification_df = classification_df.set_index(idx_col)
    classification_df.index = classification_df.index.astype(str)

    m = classification_df.index == term
    if m.sum() > 0:
        classification = classification_df.loc[term, col_name]
        if isinstance(classification, pd.Series):
            classification = classification.to_list()[0]

        return classification
    else:
        val = get_close_matches(term, classification_df.index)
        if len(val) > 0:
            classification = classification_df.loc[val[0], col_name]
            if isinstance(classification, pd.Series):
                classification = classification.to_list()[0]

            return classification
        else:
            return None


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
            classification.append("Non-ns")

    return np.array(classification)


def topic_classifier(
    dset,
    dset_name,
    model,
    terms,
    cotegories,
    category,
    crowdsourced_files,
):
    """Classify topics from Neurosynth based on the classification of
    its terms
    Parameters
    ----------
    dset : obejct
        NiMARE Dataset object
    TOPICS : str
        Topics name.
    term : list
        List of terms by topic.
    categories : list
        ['Functional', 'Clinical', 'Anatomical', 'Non-Specific']
    category : str
        Category for classification: 'Functional'
    crowdsourced_files: list
        List with the path to csv files witg the classified terms
    Returns
    -------
    first3terms : list
        List with top terms
    topics_classified_df :  pandas DataFrame, shape=(n_topics, n_categories + 1)
        Topics Classified.
    keep_topics : list
        List with topic to keep
    """
    THR = 0.2
    topterms = 3
    # Classify terms
    if dset_name == "neurosynth":
        print("Classify terms using crowdsourced annotation")
        terms_classified, _ = crowdsourcing_annot(terms, crowdsourced_files)
    if dset_name == "neuroquery":
        print("Classify terms using cognitive atlas")
        # TODO: cogat_annot() function.In progress
        # terms_classified = cogat_annot()

    # worg_topic_weight: extract htis from the LDA model
    worg_topic_weight = None

    # Classify topics
    print("Classifying topics")
    topics_classified = []
    classification = []
    keep_topics = []
    first3terms = []
    topics_kept = 0
    for topic in range(len(terms)):
        summatory = []
        for _category in cotegories:
            summatory.append(
                np.sum(worg_topic_weight[topic][terms_classified[topic] == _category])
            )
        topics_classified.append(summatory)

        # Get first topterms from the list terms
        # We can get this from the annotation name
        first3terms = None

        summatory_sorted = sorted(summatory)
        if summatory_sorted[-2] > summatory_sorted[-1] * THR:
            classification.append("ND")
        else:
            max_ind = summatory.index(summatory_sorted[-1])
            classification.append(cotegories[max_ind])
            if cotegories[max_ind] == category:
                keep_topics.append(topic)
                topics_kept += 1
    print("Numbers of {} topic kept after classification: {}".format(category, topics_kept))

    # Create output Dataframe
    topics_classified_df = pd.DataFrame(topics_classified, columns=cotegories)
    topics_classified_df.index.name = "topics"
    topics_classified_df["Classification"] = classification
    topics_classified_df.insert(0, "first3terms", first3terms)

    return first3terms, topics_classified_df, keep_topics


def classifier(terms, dset, model, dset_name, data_dir):
    # cotegories = ["Functional", "Clinical", "Anatomical", "Non-Specific"]
    # category = "Functional"

    # classification_fn = op.join(data_dir, f"{model}_{dset_name}_classification.csv")
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
        if not op.isfile(nq_term_class_fn):
            nq_categories_fn = op.join(
                data_dir, "raw", "data-neuroquery_version-1_termcategories.csv"
            )
            nq_categories_df = pd.read_csv(nq_categories_fn)

            feature_names = dset.annotations.columns.values
            feature_names = [
                f for f in feature_names if f.startswith("neuroquery6308_combined_tfidf")
            ]
            features = [f.split("__")[-1] for f in feature_names]

            nq_term_class_df = neuroquery_annot(features, ns_term_class_df, nq_categories_df)
            nq_term_class_df.to_csv(nq_term_class_fn)
        else:
            nq_term_class_df = pd.read_csv(nq_term_class_fn, index_col="FEATURE")

        term_class_df = nq_term_class_df.copy()

    if model == "term":
        term_classified = term_classifier(terms, term_class_df)
    else:
        # term_classified = topic_classifier()
        pass

    return term_classified
