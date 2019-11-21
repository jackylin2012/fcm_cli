import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from dataset.base_dataset import BaseDataset, encode_documents

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.relpath(CURRENT_DIR, "../../"), "data", "wcai")

class WcaiDataset(BaseDataset):
    data = None
    MIN_DF = 0.01
    MAX_DF = 0.8

    def load_data(self, vocab_size, window_size):
        if WcaiDataset.data is not None:
            return WcaiDataset.data
        # TODO: add override flag
        file_name = os.path.join(DATA_DIR, "wcai_%d.pkl" % window_size)
        if os.path.exists(file_name):
            WcaiDataset.data = pickle.load(open(file_name, "rb"))
            return WcaiDataset.data
        df_exp = pd.read_hdf(os.path.join(DATA_DIR, "df_exp_var.hdf"), "df_exp_var")

        print("Reading data...")
        df_reviews = pd.read_hdf(os.path.join(DATA_DIR, "df_select_nested_list.hdf"), key="wcai")

        print("Constructing positive/negative user-item pairs...")

        # for each item, construct a set of users that purchased the item
        positive_pairs = []
        negative_pairs = []

        # construct positive pairs
        for (productid, bvid), df_select in df_reviews.groupby(level=[0, 1]):
            purchased = df_select["purchased"].values[0]
            if purchased:
                positive_pairs.append((bvid, productid))

        # construct negative pairs
        bvids = set([p[0] for p in positive_pairs])
        for bvid in bvids:
            df_select = df_reviews.xs(bvid, level="bvid")
            for productid, df_select2 in df_select.groupby(level=0):
                # assert len(df_select2["arr_review_contentid"]) > 0
                purchased = df_select2["purchased"].values[0]
                if not purchased:
                    negative_pairs.append((bvid, productid))

        print(str(len(positive_pairs)) + " positive pairs")
        print(str(len(negative_pairs)) + " negative pairs")

        print("Constructing document-label pairs...")
        pos_documents = [df_reviews.loc[productid].loc[bvid]["arr_review_contentid"] for bvid, productid in
                         positive_pairs]
        neg_documents = [df_reviews.loc[productid].loc[bvid]["arr_review_contentid"] for bvid, productid in
                         negative_pairs]
        documents = pos_documents + neg_documents
        labels = np.array([1] * len(pos_documents) + [0] * len(neg_documents))
        documents = [[" ".join(review) for review in document] for document in documents]
        
        # explanatory vars
        expvars = []
        all_pairs = list(positive_pairs) + list(negative_pairs)
        for bvid, productid in all_pairs:
            expvars.append(df_exp.loc[productid].loc[bvid].values)
        expvars = np.array(expvars)
        print(expvars.shape)
        
        concatenated_documents = [" ".join(review_list) for review_list in documents]

        doc_train, doc_test, y_train, y_test, expvars_train, expvars_test = train_test_split(
            concatenated_documents,
            labels, expvars, test_size=0.15)

        vectorizer = CountVectorizer(min_df=WcaiDataset.MIN_DF, max_df=WcaiDataset.MAX_DF, max_features=vocab_size)
        X_train, y_train, X_test, wordcounts_train, doc_lens, valid_vocab, doc_windows_train, _ = \
            encode_documents(vectorizer, window_size, doc_train, y_train, doc_test, expvars_train)
        print(expvars_train.shape)
        print(expvars_test.shape)

        WcaiDataset.data = {
            "doc_windows": doc_windows_train,
            "word_counts": wordcounts_train,
            "doc_lens": doc_lens,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "vocab": vectorizer.get_feature_names(),
            "expvars_train": expvars_train,
            "expvars_test": expvars_test
        }
        pickle.dump(WcaiDataset.data, open(file_name, "wb"))
        return WcaiDataset.data
