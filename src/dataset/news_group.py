import os
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from dataset.base_dataset import BaseDataset, encode_documents

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir)), "data", "news_group")
MIN_DF = 0.01
MAX_DF = 0.8


class NewsDataset(BaseDataset):
    def __init__(self):
        newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, remove=("headers", "footers", "quotes"),
                                              categories=["alt.atheism", "comp.graphics"])
        newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, remove=("headers", "footers", "quotes"),
                                             categories=["alt.atheism", "comp.graphics"])
        self.doc_train = newsgroups_train.data
        self.y_train = newsgroups_train.target
        self.doc_test = newsgroups_test.data
        self.y_test = newsgroups_test.target

    def load_data(self, params):
        window_size = params["window_size"]  # context window size
        vocab_size = params["vocab_size"]  # max vocabulary size
        min_df = params.get("min_df", MIN_DF)  # min document frequency of vocabulary, defaults to MIN_DF
        max_df = params.get("max_df", MAX_DF)  # max document frequency of vocabulary, defaults to MAX_DF
        # TODO: add override flag
        file_name = os.path.join(DATA_DIR, "20_news_group_window_%d_vocab_%d.pkl" % window_size)
        if os.path.exists(file_name):
            NewsDataset.data = pickle.load(open(file_name, "rb"))
            return NewsDataset.data

        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=vocab_size)
        X_train, y_train, X_test, wordcounts_train, doc_lens, vocab, doc_windows_train, _ = \
            encode_documents(vectorizer, window_size, self.doc_train, self.y_train, self.doc_test)

        data = {
            "doc_windows": doc_windows_train,
            "word_counts": wordcounts_train,
            "doc_lens": doc_lens,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": self.y_test,
            "vocab": vocab
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        pickle.dump(data, open(file_name, "wb"))
        return data
