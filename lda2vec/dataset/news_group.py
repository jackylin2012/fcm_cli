from __future__ import unicode_literals

import sys

import numpy as np
import spacy
import pickle
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from dataset.base_dataset import BaseDataset, get_windows

sys.path.append("../")


def preprocess_documents(documents):
    nlp = spacy.load("en")
    documents = [' '.join(d.split()) for d in documents]
    documents = [nlp(d, disable=["parse", "tag", "ner"])
                 if len(d) > 0 else nlp(u"") for d in documents]
    documents = [" ".join([w.lemma_ for w in d if w.is_alpha and len(w) > 2 and not w.is_stop])
                 for d in documents]
    return documents


class NewsDataset(BaseDataset):
    data = None

    def load_data(self, window_size):
        if NewsDataset.data is not None:
            return NewsDataset.data
        # TODO: add override flag
        file_name = "20_news_group_window_%d.pkl" % window_size
        if os.path.exists("20_news_group.pkl"):
            NewsDataset.data = pickle.load(open(file_name, "rb"))
            return NewsDataset.data
        newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, remove=("headers", "footers", "quotes"),
                                              categories=["alt.atheism", "comp.graphics"])
        newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, remove=("headers", "footers", "quotes"),
                                             categories=["alt.atheism", "comp.graphics"])
        documents_train = newsgroups_train.data
        y_train = newsgroups_train.target
        documents_test = newsgroups_test.data
        y_test = newsgroups_test.target
        vectorizer = CountVectorizer(min_df=0.01, max_df=0.8)
        X_train = vectorizer.fit_transform(documents_train)
        X_test = vectorizer.transform(documents_test)
        analyze = vectorizer.build_analyzer()
        print(X_train.shape)
        print(X_test.shape)
        encoded_documents_train = [[vectorizer.vocabulary_.get(word)
                                    for word in analyze(document)
                                    if vectorizer.vocabulary_.get(word) is not None]
                                   for document in documents_train]
        encoded_documents_test = [[vectorizer.vocabulary_.get(word)
                                   for word in analyze(document)
                                   if vectorizer.vocabulary_.get(word) is not None]
                                  for document in documents_test]
        
        document_lengths_train = [len(d) for d in encoded_documents_train]
        documents_train = [d for d, l in zip(documents_train, document_lengths_train) if l >= window_size + 1]
        y_train = np.array([y for y, l in zip(y_train, document_lengths_train) if l >= window_size + 1])
        document_lengths_test = [len(d) for d in encoded_documents_test]
        documents_test = [d for d, l in zip(documents_test, document_lengths_test) if l >= window_size + 1]
        y_test = np.array([y for y, l in zip(y_test, document_lengths_test) if l >= window_size + 1])
        
        print(len(documents_train))
        print(len(documents_train))
        print(y_train.shape)
        print(y_test.shape)
        
        vectorizer = CountVectorizer(min_df=0.01, max_df=0.8)
        X_train = vectorizer.fit_transform(documents_train).toarray()
        X_test = vectorizer.transform(documents_test).toarray()
        analyze = vectorizer.build_analyzer()
        print(X_train.shape)
        print(X_test.shape)
        encoded_documents_train = [[vectorizer.vocabulary_.get(word)
                                    for word in analyze(document)
                                    if vectorizer.vocabulary_.get(word) is not None]
                                   for document in documents_train]

        documents_train_windows = get_windows(encoded_documents_train, y_train, window_size=window_size)
        doclens_train = np.array([len(doc) for doc in encoded_documents_train])
        wordcounts_train = X_train.sum(axis=0)

        NewsDataset.data = {
            "doc_windows": documents_train_windows,
            "word_counts": wordcounts_train,
            "doc_lens": doclens_train,
            "ndocs": len(encoded_documents_train),
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "vocab": vectorizer.get_feature_names()
        }
        pickle.dump(NewsDataset.data, open(file_name, "wb"))
        return NewsDataset.data
