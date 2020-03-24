import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from dataset.base_dataset import BaseDataset, get_data_dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir)), "data", "wcai")
MIN_DF = 0.01
MAX_DF = 0.8
TEST_RATIO = 0.15


class WcaiDataset(BaseDataset):
    def __init__(self):
        print("Reading data...")
        df = pd.read_csv(os.path.join(DATA_DIR, "wcai.csv"), encoding="utf-8")
        labels = df["labels"].values
        concatenated_documents = df["docs"].values
        # explanatory vars
        del df["labels"]
        del df["docs"]
        expvars = df.values

        self.doc_train, self.doc_test, self.y_train, self.y_test, self.expvars_train, self.expvars_test = \
            train_test_split(concatenated_documents, labels, expvars, test_size=TEST_RATIO)

    def load_data(self, params):
        window_size = params["window_size"]  # context window size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        file_name = os.path.join(DATA_DIR, "wcai_%d.pkl" % window_size)
        if os.path.exists(file_name):
            WcaiDataset.data = pickle.load(open(file_name, "rb"))
            return WcaiDataset.data

        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        data = get_data_dict(vectorizer, window_size, self.doc_train, self.y_train, self.doc_test, self.y_test,
                             self.expvars_train, self.expvars_test)
        pickle.dump(data, open(file_name, "wb"))
        return data
