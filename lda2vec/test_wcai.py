#!/usr/bin/env python

import argparse

import matplotlib

from constants import *

matplotlib.use('Agg')
import numpy as np
import torch
from SLda2vec import SLda2vec
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

torch.backends.cudnn.benchmark = True

def test_wcai(args):
    WINDOW_SIZE = int(args["window_size"])
    EMBED_SIZE = int(args["embedding_size"])
    NNEGS = int(args["nnegs"])
    NTOPICS = int(args["ntopics"])
    SAVED_MODEL_FOLDER = args["output_folder"]
    lam = float(args["lambda"])
    rho = float(args["rho"])
    eta = float(args["eta"])
    nepochs = int(args["nepochs"])
    nreviews = int(args["nreviews"])
    device = int(args["device"])
    use_expvars = args["expvars"]
    predict = args["predict"]
    validate = args["validate"]
    assert not (predict and validate)

    print("[GPU]" + str(GPU))
    if GPU:
        torch.cuda.device(device)
        current_device = torch.cuda.current_device()
        print("\tActive GPU:" + str(current_device))
    if predict:
        print("Mode: PREDICT")
    elif validate:
        print("Mode: VALIDATE")
    else:
        print("Mode: TRAIN")

    print("Model folder: " + SAVED_MODEL_FOLDER)
    print("Data folder: "+ DATA_FOLDER)

    print("Loading preprocessed data...")
    windows = np.load(open(DATA_FOLDER + "windows.npy", "r"))
    encoder = pickle.load(open(DATA_FOLDER + "encoder.p", "r"))
    encoded_docs = pickle.load(open(DATA_FOLDER + "encoded_docs.p", "r"))
    labels = np.load(open(DATA_FOLDER + "document_labels.npy", "r"))
    wordcounts = np.load(open(DATA_FOLDER + "wordcounts.npy", "r"))
    docweights = np.load(open(DATA_FOLDER + "docweights.npy", "r"))
    linear_weights = np.load(open(DATA_FOLDER + "linear_weights.npy", "r"))
    if use_expvars:
        print("\tLoading explanatory variables...")
        expvars = np.load(open(DATA_FOLDER + "expvars.npy", "r"))

    vocab_size = len(encoder)
    embed_size = windows.shape[1]
    ndocs = len(encoded_docs)
    del encoded_docs

    word_vectors = np.load(open(DATA_FOLDER + "w2v.save", "r"))
    """
    w2v = gensim.models.Word2Vec.load(SAVED_MODEL_FOLDER + "w2v.save")
    w2v.init_sims(replace=True)
    word_vectors = np.zeros((vocab_size, EMBED_SIZE)).astype('float32')
    for i in encoder.values():
        try:
            word_vectors[i] = w2v.wv[str(i)]
        except:
            continue
    """

    doc_topic_weights = np.load(open(DATA_FOLDER + "doc_topic_weights.npy", "r"))
    doc_topic_weights = np.log(doc_topic_weights + 1e-4)
    temperature = 7.0
    doc_topic_weights /= temperature

    l2v = SLda2vec(vocab_size=vocab_size, embedding_size=EMBED_SIZE, nepochs=nepochs, nnegs=NNEGS,
                   word_counts=wordcounts, ntopics=NTOPICS, ndocs=ndocs, lam=lam, rho=rho, eta=eta,
                   doc_weights=docweights, doc_topic_weights=doc_topic_weights, word_vectors=word_vectors,
                   theta=linear_weights, gpu=GPU)
    if predict:
        l2v.predict(windows, batch_size=1024*55, lr=0.01, savefolder=SAVED_MODEL_FOLDER)
    elif validate:
        l2v.predict(windows, batch_size=1024*55, lr=0.01, savefolder=SAVED_MODEL_FOLDER, validation=True)
    else:
        l2v.fit(windows, batch_size=1024*40, lr=0.01, savefolder=SAVED_MODEL_FOLDER)
