from __future__ import unicode_literals

import sys

import numpy as np
import spacy
import pickle
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append("../")

# %%
WINDOW_SIZE = 4
EMBED_SIZE = 50
NTOPICS = 2

SAVED_DATA = "20_news_group.pkl"


# TODO: move to util
def get_windows(encoded_docs, labels, window_size=4):
    half_window = window_size // 2
    windows = []
    for i in range(len(encoded_docs)):
        # for i in tqdm(range(len(encoded_docs)), desc="docs"):
        concatenated_doc = encoded_docs[i]
        label = labels[i]

        doc_len = len(concatenated_doc)

        for j in range(doc_len):
            target = concatenated_doc[j]

            if j < half_window:
                left_context = concatenated_doc[0:j]
                remaining = half_window - j
                right_context = concatenated_doc[j + 1:min(j + half_window + 1 + remaining, doc_len)]

            elif doc_len - j - 1 < half_window:
                right_context = concatenated_doc[j + 1:doc_len]
                remaining = half_window - (doc_len - j - 1)
                left_context = concatenated_doc[max(0, j - half_window - remaining):j]

            else:
                left_context = concatenated_doc[max(0, j - half_window):j]
                right_context = concatenated_doc[j + 1:min(j + half_window + 1, doc_len)]

            windows.append([i, target] + left_context + right_context + [label])

    windows_array = np.zeros((len(windows), window_size + 3), dtype=np.int)
    for i, w in enumerate(windows):
        windows_array[i, :] = w

    return windows_array


def preprocess_documents(documents):
    nlp = spacy.load("en")
    documents = [' '.join(d.split()) for d in documents]
    documents = [nlp(d, disable=["parse", "tag", "ner"])
                 if len(d) > 0 else nlp(u"") for d in documents]
    documents = [" ".join([w.lemma_ for w in d if w.is_alpha and len(w) > 2 and not w.is_stop])
                 for d in documents]
    return documents


TRAIN_SIZE = 200
TEST_SIZE = 100

def load_data():
    if os.path.exists(SAVED_DATA):
        return pickle.load(open(SAVED_DATA, "rb"))
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, remove=("headers", "footers", "quotes"),
                                          categories=["alt.atheism", "comp.graphics"])
    newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, remove=("headers", "footers", "quotes"),
                                         categories=["alt.atheism", "comp.graphics"])
    documents_train = newsgroups_train.data[:TRAIN_SIZE]
    y_train = newsgroups_train.target[:TRAIN_SIZE]
    documents_test = newsgroups_test.data[:TEST_SIZE]
    y_test = newsgroups_test.target[:TEST_SIZE]
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
    # %%
    document_lengths_train = [len(d) for d in encoded_documents_train]
    documents_train = [d for d, l in zip(documents_train, document_lengths_train) if l >= WINDOW_SIZE + 1]
    y_train = np.array([y for y, l in zip(y_train, document_lengths_train) if l >= WINDOW_SIZE + 1])
    document_lengths_test = [len(d) for d in encoded_documents_test]
    documents_test = [d for d, l in zip(documents_test, document_lengths_test) if l >= WINDOW_SIZE + 1]
    y_test = np.array([y for y, l in zip(y_test, document_lengths_test) if l >= WINDOW_SIZE + 1])
    # %%
    print(len(documents_train))
    print(len(documents_train))
    print(y_train.shape)
    print(y_test.shape)
    # %%
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

    # %%
    documents_train_windows = get_windows(encoded_documents_train, y_train, window_size=WINDOW_SIZE)
    # %%

    doclens_train = np.array([len(doc) for doc in encoded_documents_train])
    docweights_train = 1.0 / np.log(doclens_train)

    vocab_size = len(vectorizer.vocabulary_)
    wordcounts_train = np.zeros(vocab_size)
    for idx, doc in enumerate(encoded_documents_train):
        for word in doc:
            wordcounts_train[word] += 1

    data = {
        "vocab_size": vocab_size,
        "doc_windows": documents_train_windows,
        "word_counts": wordcounts_train,
        "doc_weights": docweights_train,
        "ndocs": len(encoded_documents_train),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }
    pickle.dump(data, open(SAVED_DATA, "wb"))
    return data
