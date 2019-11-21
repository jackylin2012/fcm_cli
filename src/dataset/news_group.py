import os
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from dataset.base_dataset import BaseDataset, encode_documents

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.relpath(CURRENT_DIR, "../../"), "data", "news_group")


class NewsDataset(BaseDataset):
    data = None
    MIN_DF = 0.01
    MAX_DF = 0.8

    def load_data(self, vocab_size, window_size):
        if NewsDataset.data is not None:
            return NewsDataset.data
        # TODO: add override flag
        file_name = os.path.join(DATA_DIR, "20_news_group_window_%d.pkl" % window_size)
        if os.path.exists(file_name):
            NewsDataset.data = pickle.load(open(file_name, "rb"))
            return NewsDataset.data
        newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, remove=("headers", "footers", "quotes"),
                                              categories=["alt.atheism", "comp.graphics"])
        newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, remove=("headers", "footers", "quotes"),
                                             categories=["alt.atheism", "comp.graphics"])
        doc_train = newsgroups_train.data
        y_train = newsgroups_train.target
        doc_test = newsgroups_test.data
        y_test = newsgroups_test.target
        vectorizer = CountVectorizer(min_df=NewsDataset.MIN_DF, max_df=NewsDataset.MAX_DF, max_features=vocab_size)
        X_train, y_train, X_test, wordcounts_train, doc_lens, valid_vocab, doc_windows_train, _ = \
            encode_documents(vectorizer, window_size, doc_train, y_train, doc_test)

        NewsDataset.data = {
            "doc_windows": doc_windows_train,
            "word_counts": wordcounts_train,
            "doc_lens": doc_lens,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "vocab": vectorizer.get_feature_names()
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        pickle.dump(NewsDataset.data, open(file_name, "wb"))
        return NewsDataset.data
