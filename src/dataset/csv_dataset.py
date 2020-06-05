import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from dataset.base_dataset import BaseDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = ""
DATA_NAME = ""
MIN_DF = 0.01
MAX_DF = 0.8
TEST_RATIO = 0.15
doc_train, doc_test, y_train, y_test, expvars_train, expvars_test = None, None, None, None, None, None


class CSVDataset(BaseDataset):
    def __init__(self, file_path, text_col, label_col):
        global DATA_DIR, DATA_NAME, doc_train, doc_test, y_train, y_test, expvars_train, expvars_test
        if doc_train is None:
            DATA_DIR = os.path.dirname(file_path)
            DATA_NAME = os.path.basename(file_path)
            df = pd.read_csv(file_path, encoding="utf-8")
            labels = df[label_col].to_numpy()
            concatenated_documents = df[text_col].to_numpy()
            # explanatory vars
            del df[label_col]
            del df[text_col]
            expvars = df.values
            doc_train, doc_test, y_train, y_test, expvars_train, expvars_test = \
                train_test_split(concatenated_documents, labels, expvars, test_size=TEST_RATIO)
        super().__init__(doc_train, doc_test, y_train, y_test, expvars_train, expvars_test)

    def get_data_filename(self, params):
        window_size = params["window_size"]  # context window size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        return os.path.join(DATA_DIR, "%s_w%d_min%.0E_max%.0E.pkl" % (DATA_NAME, window_size, min_df, max_df))

    def load_data(self, params):
        window_size = params["window_size"]  # context window size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        return self.get_data_dict(self.get_data_filename(params), vectorizer, window_size)
