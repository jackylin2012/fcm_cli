#!/usr/bin/env python

SEED = 42
NJOBS = 4
GPU = True
DOC_FILE = "data/doc.pickle"
DOC_LABEL_FILE = "data/doc.pickle"
DATA_FOLDER = "./data/labelled_documents/"

BETA = 0.75 # power for SGNS
ETA = 0.4 # gradient noise https://arxiv.org/abs/1511.06807

TOPIC_WEIGHTS_DECAY = 1e-2
CLASS_WEIGHTS_DECAY = 1e-2
GRAD_CLIP = 5.0

PIVOTS_DROPOUT = 0.5
DOC_VECS_DROPOUT = 0.25
DOC_WEIGHTS_INIT = 0.1

EPS = 1e-9
