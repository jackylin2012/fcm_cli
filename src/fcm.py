import logging
import os

import numpy as np
import pyLDAvis
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from scipy.spatial.distance import cdist
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score
from torch.nn import Parameter

from constants import *
from util import AliasMultinomial

torch.manual_seed(SEED)
np.random.seed(SEED)


class FocusedConceptMiner(nn.Module):

    def __init__(self, out_dir, embed_size=300, nnegs=15, ntopics=25,
                 lam=100.0, rho=100.0, eta=1.0, word_counts=None, doc_lens=None, doc_topic_weights=None,
                 word_vectors=None, theta=None, gpu=None, inductive=True,
                 X_test=None, y_test=None, X_train=None, y_train=None, doc_windows=None,
                 vocab=None, expvars_train=None, expvars_test=None,
                 file_log=False):
        """

        Parameters
        ----------
        embed_size
        nepochs
        nnegs
        word_counts
        ntopics
        lam
        rho
        eta
        doc_weights
        doc_topic_weights
        word_vectors
        expvars_train
        expvars_test
        theta
        gpu
        inductive
        X_test
        y_test
        X_train
        y_train
        doc_windows
        """
        super(FocusedConceptMiner, self).__init__()
        ndocs = X_train.shape[0]
        vocab_size = X_train.shape[1]
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.embed_size = embed_size
        self.nnegs = nnegs
        self.ntopics = ntopics
        self.lam = lam
        self.rho = rho
        self.eta = eta
        self.alpha = 1.0 / ntopics
        self.expvars_train = expvars_train
        self.expvars_test = expvars_test
        self.inductive = inductive
        self.X_train = X_train
        if self.X_train is not None:
            self.X_train = torch.FloatTensor(self.X_train)
        assert not (self.inductive and self.X_train is None)
        self.y_train = y_train
        if self.y_train is not None:
            self.y_train = torch.FloatTensor(self.y_train)
        self.X_test = X_test
        if self.X_test is not None:
            self.X_test = torch.FloatTensor(self.X_test)
        self.y_test = y_test
        if self.y_test is not None:
            self.y_test = torch.FloatTensor(self.y_test)

        self.train_dataset = PermutedSubsampledCorpus(doc_windows)

        if doc_lens is None:
            self.docweights = np.ones(ndocs, dtype=np.float)
        else:
            self.docweights = 1.0 / np.log(doc_lens)
            self.doc_lens = doc_lens
        self.docweights = torch.FloatTensor(self.docweights)

        if expvars_train is not None:
            self.expvars_train = torch.FloatTensor(expvars_train)
        if expvars_test is not None:
            self.expvars_test = torch.FloatTensor(expvars_test)

        # word embedding
        self.embedding_i = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embed_size,
                                        sparse=False)
        if word_vectors is not None:
            self.embedding_i.weight.data = torch.FloatTensor(word_vectors)

        # regular embedding for topics (never indexed so not sparse)
        self.embedding_t = nn.Parameter(torch.FloatTensor(ortho_group.rvs(embed_size)[0:ntopics]))
        if file_log:
            logging.basicConfig(filename="fcm.log", format='%(asctime)s : %(levelname)s : %(message)s')
        else:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
        self.logger = logging.getLogger(__name__)

        # embedding for per-document topic weights
        if self.inductive:
            hidden_layer_size = HIDDEN_LAYER_DIM
            num_hidden_layers = NUM_LAYERS
            weight_generator_network = []
            if num_hidden_layers > 0:
                # input layer
                weight_generator_network.extend([torch.nn.Linear(vocab_size,
                                                                 hidden_layer_size),
                                                 torch.nn.Tanh(),
                                                 torch.nn.Dropout(0.01)])
                # hidden layers
                for h in range(num_hidden_layers):
                    weight_generator_network.extend([torch.nn.Linear(hidden_layer_size,
                                                                     hidden_layer_size),
                                                     torch.nn.Tanh(),
                                                     torch.nn.Dropout(0.01)])
                # output layer
                weight_generator_network.append(torch.nn.Linear(hidden_layer_size,
                                                                ntopics))
            else:
                weight_generator_network.append(torch.nn.Linear(vocab_size,
                                                                ntopics))
            self.doc_topic_weights = torch.nn.Sequential(*weight_generator_network)
            # self.doc_topic_weights = nn.Linear(vocab_size, ntopics)
        else:
            self.doc_topic_weights = nn.Embedding(num_embeddings=ndocs,
                                                  embedding_dim=ntopics,
                                                  sparse=False)
            if doc_topic_weights is not None:
                self.doc_topic_weights.weight.data = torch.FloatTensor(doc_topic_weights)

        # explanatory variables
        if expvars_train is not None:
            # TODO: add assert shape
            nexpvars = expvars_train.shape[1]
            self.theta = Parameter(torch.FloatTensor(ntopics + nexpvars + 1))  # +1 for bias
            if theta is not None:
                nvars = theta.shape[1]  # 1 + ntopics
                self.theta.data[:nvars] = torch.FloatTensor(theta)
        else:

            if theta is not None:
                self.theta = Parameter(torch.FloatTensor(theta))
            else:
                self.theta = Parameter(torch.FloatTensor(ntopics + 1))  # + 1 for bias

        # enable gradients (True by default, just confirming)
        self.embedding_i.weight.requires_grad = True
        self.embedding_t.requires_grad = True
        # self.doc_topic_weights.weight.requires_grad = True
        self.theta.requires_grad = True

        # weights for negative sampling
        wf = np.power(word_counts, BETA)  # exponent from word2vec paper
        self.word_counts = word_counts
        wf = wf / np.sum(wf)  # convert to probabilities
        self.multinomial = AliasMultinomial(wf, self.device)
        self.weights = torch.FloatTensor(wf)
        self.vocab = vocab
        # dropout
        self.dropout1 = nn.Dropout(PIVOTS_DROPOUT)
        self.dropout2 = nn.Dropout(DOC_VECS_DROPOUT)
        if torch.cuda.is_available():
            if gpu:
                self.device = "cuda:%d" % gpu
            else:
                self.device = "cuda:0"
        else:
            self.device = 'cpu'

    def forward(self, doc, target, contexts, labels, per_doc_loss=None):
        """
        Args:
            doc:        [batch_size,1] LongTensor of document indices
            target:     [batch_size,1] LongTensor of target (pivot) word indices
            contexts:   [batchsize,window_size] LongTensor of convext word indices
            labels:     [batchsize,1] LongTensor of document labels

            All arguments are tensors wrapped in Variables.
        """
        batch_size, window_size = contexts.size()

        # reweight loss by document length
        w = autograd.Variable(self.docweights[doc.data])
        w /= w.sum()
        w *= w.size(0)

        # construct document vector = weighted linear combination of topic vectors
        if self.inductive:
            doc_topic_weights = self.doc_topic_weights(self.X_train[doc])
        else:
            doc_topic_weights = self.doc_topic_weights(doc)
        doc_topic_probs = doc_topic_weights
        doc_topic_probs = doc_topic_probs.unsqueeze(1)  # (batches, 1, T)
        topic_embeddings = self.embedding_t.expand(batch_size, -1, -1)  # (batches, T, E)
        doc_vector = torch.bmm(doc_topic_probs, topic_embeddings)  # (batches, 1, E)
        doc_vector = doc_vector.squeeze(dim=1)  # (batches, E)
        doc_vector = self.dropout2(doc_vector)

        # sample negative word indices for negative sampling loss; approximation by sampling from the whole vocab
        if self.device == "cpu":
            nwords = torch.multinomial(self.weights, batch_size * window_size * self.nnegs,
                                       replacement=True).view(batch_size, -1)
            nwords = autograd.Variable(nwords)
        else:
            nwords = self.multinomial.draw(batch_size * window_size * self.nnegs)
            nwords = autograd.Variable(nwords).view(batch_size, window_size * self.nnegs)

        # compute word vectors
        ivectors = self.dropout1(self.embedding_i(target))  # column vector
        ovectors = self.embedding_i(contexts)  # (batches, window_size, E)
        nvectors = self.embedding_i(nwords).neg()  # row vector

        # construct "context" vector defined by lda2vec
        context_vectors = doc_vector + ivectors
        context_vectors = context_vectors.unsqueeze(2)  # column vector, batch needed for bmm

        # compose negative sampling loss
        oloss = torch.bmm(ovectors, context_vectors).squeeze(dim=2).sigmoid().clamp(min=EPS).log().sum(1)
        nloss = torch.bmm(nvectors, context_vectors).squeeze(dim=2).sigmoid().clamp(min=EPS).log().sum(1)
        negative_sampling_loss = (oloss + nloss).neg()
        negative_sampling_loss *= w  # downweight loss for each document
        if per_doc_loss is not None:
            per_doc_loss[doc] += negative_sampling_loss.data
        negative_sampling_loss = negative_sampling_loss.mean()  # mean over the batch

        # compose dirichlet loss
        doc_topic_probs = doc_topic_probs.squeeze(dim=1)  # (batches, T)
        doc_topic_probs = doc_topic_probs.clamp(min=EPS)
        dirichlet_loss = doc_topic_probs.log().sum(1)  # (batches, 1)
        dirichlet_loss *= self.lam * (1.0 - self.alpha)
        dirichlet_loss *= w  # downweight loss for each document
        if per_doc_loss is not None:
            per_doc_loss[doc] += dirichlet_loss.data
        dirichlet_loss = dirichlet_loss.mean()  # mean over the entire batch

        ones = torch.ones((batch_size, 1)).to(self.device)
        doc_topic_probs = torch.cat((ones, doc_topic_probs), dim=1)

        # expand doc_topic_probs vector with explanatory variables
        if self.expvars_train is not None:
            doc_topic_probs = torch.cat((doc_topic_probs, self.expvars_train[doc, :]),
                                        dim=1)
        # compose prediction loss
        # [batch_size] = torch.matmul([batch_size, ntopics], [ntopics])
        pred_weight = torch.matmul(doc_topic_probs, self.theta)
        pred_loss = F.binary_cross_entropy_with_logits(pred_weight, labels,
                                                       weight=w, reduction='none')
        pred_loss *= self.rho
        if per_doc_loss is not None:
            per_doc_loss[doc] += pred_loss.data
        pred_loss = pred_loss.mean()

        # compose diversity loss
        #   1. First compute \sum_i \sum_j log(sigmoid(T_i, T_j))
        #   2. Then compute \sum_i log(sigmoid(T_i, T_i))
        #   3. Loss = (\sum_i \sum_j log(sigmoid(T_i, T_j)) - \sum_i log(sigmoid(T_i, T_i)) )
        #           = \sum_i \sum_{j > i} log(sigmoid(T_i, T_j))
        div_loss = torch.mm(self.embedding_t,
                            torch.t(self.embedding_t)).neg().sigmoid().clamp(min=EPS).log().sum() \
                   - (self.embedding_t * self.embedding_t).neg().sigmoid().clamp(min=EPS).log().sum()
        div_loss /= 2.0  # taking care of duplicate pairs T_i, T_j and T_j, T_i
        div_loss = div_loss.repeat(batch_size)
        div_loss *= w  # downweight by document lengths
        div_loss *= self.eta
        if per_doc_loss is not None:
            per_doc_loss[doc] += div_loss.data
        div_loss = div_loss.mean()  # mean over the entire batch

        return negative_sampling_loss, dirichlet_loss, pred_loss, div_loss

    def fit(self, lr=0.01, nepochs=200, batch_size=10, weight_decay=0.01, grad_clip=5, save_epochs=10):
        """
        Train the FCM model

        Parameters
        ----------
        lr : float
            Learning rate
        nepochs : int
            Number of training epochs
        batch_size : int
            Batch size
        weight_decay : float
            Adam optimizer weight decay (L2 penalty)
        grad_clip : float
            Maximum gradients magnitude. Gradients will be clipped within the range [-grad_clip, grad_clip]
        save_epochs : int
            The number of epochs in between saving the model weights

        Returns
        -------
        metrics : ndarray, shape (n_epochs, 6)
            Training metrics from each epoch including: total_loss, avg_sgns_loss, avg_diversity_loss, avg_pred_loss,
            avg_diversity_loss, train_auc, test_auc
        """
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size, shuffle=True,
                                                       num_workers=4, pin_memory=True,
                                                       drop_last=False)
        self.to(self.device)
        train_loss_file = open(os.path.join(self.out_dir, "train_loss.txt"), "w")
        train_loss_file.write("total_loss, avg_sgns_loss, avg_diversity_loss, avg_pred_loss, "
                              "avg_diversity_loss，train_auc, test_auc\n")

        # SGD generalizes better: https://arxiv.org/abs/1705.08292
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        nwindows = len(self.train_dataset)
        metrics = []
        for epoch in range(nepochs):
            total_sgns_loss = 0.0
            total_dirichlet_loss = 0.0
            total_pred_loss = 0.0
            total_diversity_loss = 0.0

            self.train()
            for batch in train_dataloader:
                loss, sgns_loss, dirichlet_loss, pred_loss, div_loss = self.calculate_loss(batch)

                optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                for p in self.parameters():
                    if p.requires_grad:
                        p.grad = p.grad.clamp(min=-grad_clip, max=grad_clip)

                optimizer.step()

                nsamples = batch.size(0)

                total_sgns_loss += sgns_loss.data * nsamples
                total_dirichlet_loss += dirichlet_loss.data * nsamples
                total_pred_loss += pred_loss.data * nsamples
                total_diversity_loss += div_loss.data * nsamples

            train_auc = self.calculate_auc("Train", self.X_train, self.y_train, self.expvars_train)
            test_auc = self.calculate_auc("Test", self.X_test, self.y_test, self.expvars_test)

            total_loss = (total_sgns_loss + total_dirichlet_loss + total_pred_loss + total_diversity_loss) / nwindows
            avg_sgns_loss = total_sgns_loss / nwindows
            avg_dirichlet_loss = total_dirichlet_loss / nwindows
            avg_pred_loss = total_pred_loss / nwindows
            avg_diversity_loss = total_dirichlet_loss / nwindows
            self.logger.info("epoch %d/%d:" % (epoch, nepochs))
            self.logger.info("Total loss: %.4f" % total_loss)
            self.logger.info("SGNS loss: %.4f" % avg_sgns_loss)
            self.logger.info("Dirichlet loss: %.4f" % avg_dirichlet_loss)
            self.logger.info("Prediction loss: %.4f" % avg_pred_loss)
            self.logger.info("Diversity loss: %.4f" % avg_diversity_loss)
            train_loss_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" %
                                  (total_loss, avg_sgns_loss, avg_diversity_loss, avg_pred_loss,
                                   avg_diversity_loss, train_auc, test_auc))
            train_loss_file.flush()
            metrics.append([total_loss, avg_sgns_loss, avg_diversity_loss, avg_pred_loss,
                            avg_diversity_loss, train_auc, test_auc])
            if (epoch + 1) % save_epochs == 0:
                torch.save(self.state_dict(), os.path.join(self.out_dir, str(epoch + 1) + ".slda2vec.pytorch"))

        torch.save(self.state_dict(), os.path.join(self.out_dir, "slda2vec.pytorch"))
        return np.array(metrics)

    def calculate_auc(self, split, X, y, expvars):
        if self.expvars_test is None:
            y_pred = self.predict_proba(X).cpu().detach().numpy()
        else:
            y_pred = self.predict_proba(X, expvars).cpu().detach().numpy()
        auc = roc_auc_score(y, y_pred)
        self.logger.info("%s AUC: %.4f" % (split, auc))
        return auc

    def calculate_loss(self, batch, per_doc_loss=None):
        batch = autograd.Variable(torch.LongTensor(batch))
        batch = batch.to(self.device)
        doc = batch[:, 0]
        iword = batch[:, 1]
        owords = batch[:, 2:-1]
        labels = batch[:, -1].float()

        sgns_loss, dirichlet_loss, pred_loss, div_loss = self(doc, iword, owords,
                                                              labels, per_doc_loss)
        loss = sgns_loss + dirichlet_loss + pred_loss + div_loss
        return loss, sgns_loss, dirichlet_loss, pred_loss, div_loss

    # TODO: only applicable to inductive, figure out what to do for non-inductive
    def predict_proba(self, count_matrix, expvars=None):
        assert self.inductive
        with torch.no_grad():
            batch_size = count_matrix.size(0)
            doc_topic_weights = self.doc_topic_weights(count_matrix)
            doc_topic_probs = F.softmax(doc_topic_weights, dim=1)  # convert to probabilities

            ones = torch.ones((batch_size, 1)).to(self.device)
            doc_topic_probs = torch.cat((ones, doc_topic_probs), dim=1)

            if expvars is not None:
                expvars = expvars
                doc_topic_probs = torch.cat((doc_topic_probs, expvars), dim=1)

            pred_weight = torch.matmul(doc_topic_probs, self.theta)
            pred_proba = pred_weight.sigmoid()
        return pred_proba

    def visualize(self):
        with torch.no_grad():
            doc_topic_weights = self.doc_topic_weights(self.X_train)
            # [n_docs, n_topics]
            doc_topic_probs = F.softmax(doc_topic_weights, dim=1)  # convert to probabilities
            # [n_topics, vocab_size]
            topic_word_dists = torch.matmul(doc_topic_probs.transpose(0, 1), self.X_train)
            vis_data = pyLDAvis.prepare(topic_term_dists=topic_word_dists.data.numpy(),
                                        doc_topic_dists=doc_topic_probs.data.numpy(),
                                        doc_lengths=self.doc_lens, vocab=self.vocab, term_frequency=self.word_counts)
            pyLDAvis.save_html(vis_data, os.path.join(self.out_dir, "visualization.html"))

    # TODO: add filtering such as pos and tf
    def get_concept_words(self, top_k=10, concept_metric='dot'):
        concept_embed = self.embedding_t.data.numpy()
        word_embed = self.embedding_i.weight.data.numpy()
        if concept_metric == 'dot':
            dist = -np.matmul(concept_embed, np.transpose(word_embed, (1, 0)))
        else:
            dist = cdist(concept_embed, word_embed, metric=concept_metric)
        nearest_words = np.argsort(dist, axis=1)[:, :top_k]  # indices of words with min cosine distance
        for j in range(self.ntopics):
            topic_words = ' '.join([self.vocab[i] for i in nearest_words[j, :]])
            # TODO: write to result file
            self.logger.info('topic %d: %s' % (j + 1, topic_words))


class PermutedSubsampledCorpus(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.data = windows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
