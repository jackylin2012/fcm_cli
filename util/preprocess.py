#!/usr/bin/env python

from __future__ import unicode_literals
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
from tqdm import tqdm

def generate_windows(documents, window_size, labels=None, expvars=None, use_spacy=True):
    """
        Returns:
            windows: A numpy matrix, each row containing the values:
                    [document index, target word index, context word index]
            vectorizer: A sklearn vectorizer containing the word-index
                        mappings in vectorizer.vocabulary_
    """
    assert window_size % 2 == 0

    # use spacy library to cleanup the documents
    if use_spacy:
        nlp = spacy.load("en")
        # remove redundant spaces
        documents = [' '.join(d.split()) for d in documents]
        documents = [nlp(d, tag=True, parse=False, entity=False) for d in documents]
        # only keep the base form of the words that are alphabetical, length > 2, and not stop words
        documents = [" ".join([w.lemma_ for w in d if w.is_alpha and len(w) > 2 and not w.is_stop])
                     for d in documents]

    concatenated_documents = [" ".join(document) for document in documents]

    # exclude words with document frequency < 0.0001, or > 0.6
    vectorizer = CountVectorizer(min_df=0.0001, max_df=0.6)
    X = vectorizer.fit_transform(concatenated_documents)
    print("Vocabulary size:" + str(len(vectorizer.get_feature_names())))
    analyze = vectorizer.build_analyzer()

    # convert words in documents into vectors based on word counts
    encoded_docs = [[[vectorizer.vocabulary_.get(word)
                      for word in analyze(review)
                      if vectorizer.vocabulary_.get(word) is not None]
                     for review in document]
                    for document in documents]

    doc_lens = np.array([sum([len(encoded_review) for encoded_review in doc])
                         for doc in encoded_docs])
    # only keep documents with length longer than the window size
    valid_doc_lens = doc_lens >= window_size + 1
    if labels is not None:
        labels = labels[valid_doc_lens]
    if expvars is not None:
        expvars = expvars[valid_doc_lens, :]
    documents = [documents[i] for i in range(len(documents)) if valid_doc_lens[i]]
    encoded_docs = [encoded_docs[i] for i in range(len(encoded_docs)) if valid_doc_lens[i]]

    windows = []
    concatenated_encoded_docs = []
    for i in tqdm(range(len(encoded_docs)), desc="Generating windows for documents"):
        doc = encoded_docs[i]
        concatenated_doc = [word for review in doc for word in review]
        concatenated_encoded_docs.append(concatenated_doc)

        doc_len = len(concatenated_doc)
        for j in range(doc_len):
            target = concatenated_doc[j]

            if j < window_size/2:
                left_context = concatenated_doc[0:j]
                remaining = window_size/2 - j
                right_context = concatenated_doc[j+1:min(j+window_size/2+1+remaining, doc_len)]
            elif doc_len - j  - 1 < window_size/2:
                right_context = concatenated_doc[j+1:doc_len]
                remaining = window_size/2 - (doc_len - j - 1)
                left_context = concatenated_doc[max(0,j-window_size/2-remaining):j]
            else:
                left_context = concatenated_doc[max(0,j-window_size/2):j]
                right_context = concatenated_doc[j+1:min(j+window_size/2+1, doc_len)]

            if labels is not None:
                windows.append([i, target] + left_context + right_context + [labels[i]])
            else:
                windows.append([i, target] + left_context + right_context)

    if labels is not None:
        windows_array = np.zeros((len(windows), window_size + 4), dtype=np.int)
    else:
        windows_array = np.zeros((len(windows), window_size + 2), dtype=np.int)
    for i, w in enumerate(windows):
        windows_array[i,:] = w

    if labels is not None and expvars is not None:
        return windows_array, vectorizer, encoded_docs, concatenated_encoded_docs, documents, labels, expvars
    elif labels is not None:
        return windows_array, vectorizer, encoded_docs, concatenated_encoded_docs, documents, labels
    else:
        return windows_array, vectorizer, encoded_docs, documents
