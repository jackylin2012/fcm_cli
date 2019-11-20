#!/usr/bin/env python

import argparse
import logging

import pickle
import gensim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.metrics as metrics
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.preprocess import generate_windows
from tqdm import tqdm


logger = logging.getLogger()
logger.setLevel(logging.ERROR)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def preprocess_wcai(args):
    WINDOW_SIZE = int(args["window_size"])
    NTOPICS = int(args["ntopics"])
    SAVED_MODEL_FOLDER = args["output_folder"]

    try:
        os.mkdir(SAVED_MODEL_FOLDER)
    except:
        print(SAVED_MODEL_FOLDER + " exists! Not creating it again.")



    # documents: pos_documents + neg_documents, list of list of reviews (str)
    # labels: np.array 1 for pos, 0 for neg
    if os.path.exists("data/doc.pickle") and os.path.exists("data/label.pickle"):
        print("Loading saved documents/labels...")
        documents = pickle.load(open("data/doc.pickle", "r"))
        labels = np.array(pickle.load(open("data/label.pickle", "r")))
    else:
        print("Reading data...")
        df_reviews = pd.read_hdf("./data/df_select_nested_list.hdf", key="wcai")

        print("Constructing positive/negative user-item pairs...")

        # for each item, construct a set of users that purchased the item
        positive_pairs = []
        negative_pairs = []

        # construct positive pairs
        for (productid, bvid), df_select in df_reviews.groupby(level=[0, 1]):
            purchased = df_select["purchased"].values[0]
            if purchased:
                positive_pairs.append((bvid, productid))

        # construct negative pairs
        bvids = set([p[0] for p in positive_pairs])
        for bvid in bvids:
            df_select = df_reviews.xs(bvid, level="bvid")
            for productid, df_select2 in df_select.groupby(level=0):
                # assert len(df_select2["arr_review_contentid"]) > 0
                purchased = df_select2["purchased"].values[0]
                if not purchased:
                    negative_pairs.append((bvid, productid))

        print(str(len(positive_pairs)) + " positive pairs")
        print(str(len(negative_pairs)) + " negative pairs")

        print("Constructing document-label pairs...")
        pos_documents = [df_reviews.loc[productid].loc[bvid]["arr_review_contentid"] for bvid, productid in positive_pairs]
        neg_documents = [df_reviews.loc[productid].loc[bvid]["arr_review_contentid"] for bvid, productid in negative_pairs]
        documents = pos_documents + neg_documents
        labels = np.array([1] * len(pos_documents) + [0] * len(neg_documents))
        documents = [[" ".join(review) for review in document] for document in documents]


    print("Constructing windows...")
    windows, vectorizer, encoded_docs, concatenated_encoded_docs, documents2, labels2 = \
        generate_windows(documents, window_size=WINDOW_SIZE, labels=labels, use_spacy=False)
    labels2 = np.array(labels2)

    np.save(open(SAVED_MODEL_FOLDER + "/document_labels.npy", "w"), labels2)
    np.save(open(SAVED_MODEL_FOLDER + "/windows.npy", "w"), windows)
    pickle.dump(vectorizer.vocabulary_, open(SAVED_MODEL_FOLDER + "/encoder.p", "w"))
    pickle.dump(encoded_docs, open(SAVED_MODEL_FOLDER + "/encoded_docs.p", "w"))
    pickle.dump(concatenated_encoded_docs, open(SAVED_MODEL_FOLDER + '/concatenated_encoded_docs.p', "w"))
    print(str(np.sum(labels2)) + " pos of " + str(len(labels2)) + " total " +
          str(len(labels2) - np.sum(labels2)) + " neg")

    # get word counts and document lengths
    doclens = np.array([len(doc) for doc in concatenated_encoded_docs])
    docweights = 1.0/np.log(doclens)

    vocab_size = len(vectorizer.vocabulary_)
    wordcounts = np.zeros(vocab_size)
    for idx, doc in enumerate(encoded_docs):
        for word in doc:
            wordcounts[word] += 1
    np.save(open(SAVED_MODEL_FOLDER + "/wordcounts.npy", "w"), wordcounts)
    np.save(open(SAVED_MODEL_FOLDER + "/docweights.npy", "w"), docweights)

    # initialize word embedding using word2vec
    saved_model_filename = SAVED_MODEL_FOLDER + "/w2v.save"
    if os.path.isfile(saved_model_filename):
        print("Saved model exists!")
    else:
        pretrained_encoder = {}
        pretrained_vectors = np.zeros((417194, 300), dtype=float)
        if not os.path.isfile("./numberbatch-en-17.06.txt"):
            print("Download the pretrained vectors, see README.md.")
            sys.exit(-1)
        with open("./numberbatch-en-17.06.txt", "r") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                fields = line.strip().split()
                word = fields[0]
                vector = map(float, fields[1:])
                pretrained_encoder[word] = idx - 1
                pretrained_vectors[idx-1, :] = vector

        print(len(set(vectorizer.vocabulary_.keys()).intersection(set(pretrained_encoder.keys()))))
        print("intersecting words with conceptnet, of total %d" % (len(vectorizer.vocabulary_)))

        decoder = {i: w for w, i in vectorizer.vocabulary_.iteritems()}
        embedding = np.random.rand(len(vectorizer.vocabulary_), 300)
        for i in range(len(vectorizer.vocabulary_)):
            word = decoder[i]
            if not word in pretrained_encoder:
                continue
            word_idx = pretrained_encoder[word]
            word_vector = pretrained_vectors[word_idx, :]
            embedding[i, :] = word_vector

        np.save(open(saved_model_filename, "w"), embedding)
        print("Saved word2vec model!")

    """
    try:
        w2v = gensim.models.Word2Vec.load(saved_model_filename)
        print("Saved word2vec model exists!")
    except:
        print("word2vec training...")
        t = time.time()
        w2v = gensim.models.Word2Vec([[str(j) for j in doc] for doc in concatenated_encoded_docs],
                                     size=EMBED_SIZE, window=WINDOW_SIZE/2, sg=1, negative=NNEGS,
                                     iter=50, workers=84, min_count=1)
        print("done in " + str(time.time() - t) + "s")
        w2v.save(saved_model_filename)
        print("Saved word2vec model!")
    """

    # initialize topic embedding with lda
    saved_lda_model = "data/lda_model.gs"
    decoder = {i: w for w, i in vectorizer.vocabulary_.iteritems()}
    texts = [[decoder[w] for w in d] for d in concatenated_encoded_docs]
    dictionary = gensim.corpora.dictionary.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    if os.path.exists(saved_lda_model):
        print("Saved LDA model exists!")
        lda = gensim.models.ldamulticore.LdaMulticore.load(saved_lda_model)
        #coherence_per_topic = [x[1] for x in lda.top_topics(corpus=corpus, topn=10)]
        #coherence_overall = np.mean(coherence_per_topic)
        #print("Coherence =" + str(coherence_overall))
    else:
        print("LDA training...")
        best_lda = None
        best_coherence = -1.0
        for iter in range(20):
            print("\ttrial:" + str(iter))
            t = time.time()
            # lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=NTOPICS,
            #                                      id2word=dictionary,
            #                                      chunksize=10000,
            #                                      eval_every=None,
            #                                      iterations=100,
            #                                      alpha="auto")
            lda = gensim.models.ldamulticore.LdaMulticore(corpus,
                                                          num_topics=NTOPICS,
                                                          workers=84,
                                                          id2word=dictionary,
                                                          eval_every=None,
                                                          iterations=100,
                                                          chunksize=100000)
            print("Done in " + str(time.time() - t) + "s")

            coherence_per_topic = [x[1] for x in lda.top_topics(corpus=corpus, topn=10)]
            coherence_overall = np.mean(coherence_per_topic)
            print("coh: " + str(coherence_overall))
            if coherence_overall > best_coherence:
                best_lda = lda
                best_coherence = coherence_overall
            lda.save(saved_lda_model + ".{:02d}".format(iter))
        lda = best_lda
        lda.save(saved_lda_model)
        print("\tBest coherence: " + str(best_coherence))

    for i, topics in lda.show_topics(NTOPICS, formatted=False):
        print('topic', i, ':', ' '.join([t for t, _ in topics]))

    print("Computing doc-topic weights...")
    corpus_lda = lda[corpus]
    doc_topic_weights = np.zeros((len(corpus_lda), NTOPICS), dtype=np.float)
    for i in tqdm(range(len(corpus_lda))):
        topics = corpus_lda[i]
        for j, prob in topics:
            doc_topic_weights[i, j] = prob
    np.save(open(SAVED_MODEL_FOLDER + "/doc_topic_weights.npy", "w"), doc_topic_weights)

    # initialize softmax layer with logistic regression
    encoded_reviews = [r for doc in encoded_docs for r in doc]
    review_pos = [int(idx > 5) for doc in encoded_docs for idx, r in enumerate(doc)]
    review_labels = [labels2[idx] for idx, doc in enumerate(encoded_docs) for r in doc]
    review_texts = [[decoder[w] for w in d] for d in encoded_reviews]
    review_dictionary = gensim.corpora.dictionary.Dictionary(review_texts)
    review_corpus = [dictionary.doc2bow(text) for text in review_texts]

    review_corpus_lda = lda[review_corpus]
    review_topic_weights = np.zeros((len(review_corpus_lda), NTOPICS), dtype=np.float)
    for i in tqdm(range(len(review_corpus_lda))):
        topics = review_corpus_lda[i]
        for j, prob in topics:
            review_topic_weights[i, j] = prob

    print("LR classifier...")
    from sklearn.linear_model import LogisticRegression
    if os.path.exists(SAVED_MODEL_FOLDER + "/linear_weights.npy"):
        print("Saved LR classifier exists!")
    else:
        # TODO: maxpos == 1?
        maxpos = max(review_pos)
        clf_weights = np.zeros((maxpos+1, 1+NTOPICS))
        clfs = []
        datas = []
        ys = []
        for pos in range(maxpos+1):
            clf = LogisticRegression(max_iter=1000.0)
            data = np.array([review_topic_weights[i, :] for i in range(len(review_pos)) if review_pos[i] == pos])
            y = np.array([review_labels[i] for i in range(len(review_pos)) if review_pos[i] == pos])
            clf.fit(data, y)
            clf_weights[pos, :] = np.concatenate((clf.intercept_.reshape(1, -1), clf.coef_), axis=1)
            clfs.append(clf)
            datas.append(data)
            ys.append(y)
        np.save(SAVED_MODEL_FOLDER + "/linear_weights.npy", clf_weights)

        plt.figure(figsize=(10, 5))
        for pos in range(maxpos+1):
            plt.plot(range(NTOPICS + 1), clf_weights[pos, :], 'o-', label="Pos. " + str(pos+1))
        plt.xticks(range(NTOPICS+1), ["bias"] + ["Topic " + str(i) for i in range(NTOPICS)], rotation=90)
        plt.legend(ncol=5)
        plt.xlabel("Topic/bias weight, conditional on position")
        plt.ylabel("Weight")
        plt.grid()
        plt.savefig(SAVED_MODEL_FOLDER + "/logreg_weights.pdf", bbox_inches="tight")

        scores = []
        ytrue = []
        for pos in range(maxpos+1):
            scores.extend(clfs[pos].predict_proba(datas[pos])[:, 1])
            ytrue.extend(ys[pos])
        ytrue = np.array(ytrue)
        scores = np.array(scores)
        fpr, tpr, thresholds = metrics.roc_curve(ytrue, scores, pos_label=1)
        plt.plot(fpr, tpr, '-')
        plt.grid()
        plt.savefig(SAVED_MODEL_FOLDER + "/logreg_roc.pdf", bbox_inches="tight")
