import os
import pickle
import numpy as np


class BaseDataset(object):
    """An abstract class representing a Dataset containing encoded documents.

    All other datasets should subclass it. All subclasses should override
    ``load_data``, that provides the necessary attributes of the dataset.
    """

    def __init__(self, doc_train, doc_test, y_train, y_test, expvars_train=None, expvars_test=None):
        """Initializes dataset properties.

        Parameters
        ----------
        doc_train: ndarray, shape (n_train_docs,)
            Training corpus representing each document as a string, where n_train_docs is the number of documents
            in the training set.
        doc_test: ndarray, shape (n_test_docs,)
            Test corpus representing each document as a string, where n_test_docs is the number of documents
            in the test set.
        y_train: ndarray, shape (n_train_docs,)
            Binary labels in the training set, ndarray with values of 0 or 1
        y_test: ndarray, shape (n_test_docs,)
            Binary labels in the test set, ndarray with values of 0 or 1
        expvars_train [OPTIONAL] : ndarray, shape (n_train_docs, n_features)
            Extra features for making prediction in the training set
        expvars_test [OPTIONAL] : ndarray, shape (n_test_docs, n_features)
            Extra features for making prediction in the test set
        """
        assert len(doc_train) == len(y_train), "len(doc_train) = %d is not equal to len(y_train) = %d" % \
                                               (len(doc_train), len(y_train))
        if expvars_train is not None:
            assert len(expvars_train) == len(y_train), "len(expvars_train) = %d is not equal to len(y_train) = %d" % \
                                                       (len(expvars_train), len(y_train))
        assert len(doc_test) == len(y_test), "len(doc_test) = %d is not equal to len(y_test) = %d" % \
                                             (len(doc_test), len(y_test))
        if expvars_test is not None:
            assert len(expvars_test) == len(y_test), "len(expvars_test) = %d is not equal to len(y_test) = %d" % \
                                                     (len(expvars_test), len(y_test))
        self.doc_train = doc_train
        self.y_train = y_train
        self.doc_test = doc_test
        self.y_test = y_test
        self.expvars_train = expvars_train
        self.expvars_test = expvars_test

    def load_data(self, params):
        """Loads the data with extracted features

        Parameters
        ----------
        params : dict
            A dictionary of parameters. "window_size" and "vocab_size" are required

        Returns
        -------
        data : dict
            A dictionary containing the following attributes of the dataset:
            {
                "bow_train": ndarray, shape (n_train_docs, vocab_size)
                    Training corpus encoded as a bag-of-words matrix, where n_train_docs is the number of documents
                    in the training set, and vocab_size is the vocabulary size.
                "bow_test": ndarray, shape (n_test_docs, vocab_size)
                    Test corpus encoded as a matrix
                "y_train": ndarray, shape (n_train_docs,)
                    Binary labels in the training set, ndarray with values of 0 or 1.
                "y_test": ndarray, shape (n_test_docs,)
                    Binary labels in the test set, ndarray with values of 0 or 1.
                "doc_windows": ndarray, shape (n_windows, windows_size + 3)
                    Context windows constructed from bow_train. Each row represents a context window, consisting of
                    the document index of the context window, the encoded target words, the encoded context words,
                    and the document's label. This can be generated with the helper function `get_windows`.
                "vocab" : array-like, shape `vocab_size`
                    List of all the unique words in the training corpus. The order of this list corresponds
                    to the columns of the `bow_train` and `bow_test`
                "word_counts": ndarray, shape (vocab_size,)
                    The count of each word in the training documents. The ordering of these counts
                    should correspond with `vocab`.
                "doc_lens" : ndarray, shape (n_train_docs,)
                    The length of each training document
                "expvars_train" [OPTIONAL] : ndarray, shape (n_train_docs, n_features)
                    Extra features for making prediction in the training set
                "expvars_test" [OPTIONAL] : ndarray, shape (n_test_docs, n_features)
                    Extra features for making prediction in the test set
            }
        """
        raise NotImplementedError

    def get_data_dict(self, filename, vectorizer, window_size):
        """Gets the dataset dictionary object with all necessary features

        Parameters
        ----------
        filename: str
            The filename to save the dataset dictionary object
        vectorizer: Vectorizer
            A vectorizer instance for converting text to  bag-of-words representation
        window_size: int
            The size of context windows

        Returns
        -------
            data : dict
                A dictionary containing the same attributes required by `BaseDataset.load_data` method
        """
        if os.path.exists(filename):
            return pickle.load(open(filename, "rb"))
        analyze = vectorizer.build_analyzer()
        bow_train = vectorizer.fit_transform(self.doc_train).toarray()
        document_lengths_train = bow_train.sum(axis=1)
        valid_docs = document_lengths_train >= window_size + 1
        document_lengths_train = document_lengths_train[valid_docs]
        encoded_documents_train = [[vectorizer.vocabulary_.get(word)
                                    for word in analyze(document)
                                    if vectorizer.vocabulary_.get(word) is not None]
                                   for document in self.doc_train]
        encoded_documents_train = filter_list(encoded_documents_train, valid_docs)
        bow_train = bow_train[valid_docs]
        y_train = np.array(self.y_train)[valid_docs]
        bow_test = vectorizer.transform(self.doc_test).toarray()
        if self.expvars_train is not None:
            expvars_train = self.expvars_train[valid_docs]
        else:
            expvars_train = None
        wordcounts_train = bow_train.sum(axis=0)
        doc_windows_train = get_windows(encoded_documents_train, y_train, window_size=window_size)
        data = {
            "doc_windows": doc_windows_train,
            "word_counts": wordcounts_train,
            "doc_lens": document_lengths_train,
            "bow_train": bow_train,
            "y_train": y_train,
            "bow_test": bow_test,
            "y_test": self.y_test,
            "vocab": vectorizer.get_feature_names(),
            "expvars_train": expvars_train,
            "expvars_test": self.expvars_test
        }
        pickle.dump(data, open(filename, "wb"))
        return data


def filter_list(original, include):
    return [original[i] for i in range(len(include)) if include[i]]


def get_windows(encoded_docs, labels, window_size):
    """
    Generate context windows from the encoded document

    Parameters
    ----------
    encoded_docs : iterable of iterable
        List of encoded documents which are list of encoded words
    labels : ndarray
        Binary labels of the encoded documents, ndarray with values of 0 or 1.
    window_size : int
        The size of context window for training the word embedding

    Returns
    -------
    doc_windows: ndarray, shape (n_windows, windows_size + 3)
        Context windows constructed from bow_train. Each row represents a context window, consisting of
        the document index of the context window, the encoded target words, the encoded context words,
        and the document's label.
    """
    half_window = window_size // 2
    windows = []
    for i in range(len(encoded_docs)):
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
            w = [i, target] + left_context + right_context + [label]
            if len(w) != window_size + 3:
                raise ValueError("j=%d, left_context=%s, right_context=%s, w=%s" % (j, left_context, right_context, w))
            windows.append([i, target] + left_context + right_context + [label])

    windows_array = np.zeros((len(windows), window_size + 3), dtype=np.int)
    for i, w in enumerate(windows):
        windows_array[i, :] = w

    return windows_array
