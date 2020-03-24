#!/usr/bin/env python
import os
import sys

import click

from fcm import FocusedConceptMiner
from toolbox.helper_functions import get_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group()
def fcm():
    pass


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.argument('dataset')
@click.argument('out-dir', type=click.Path(file_okay=False))
@click.option('--nconcepts', default=5, help="No. of concepts")
@click.option('--embed-dim', default=50, help="The size of each word/concept embedding vector")
@click.option('--vocab-size', default=10000, help="Maximum vocabulary size")
@click.option('--nnegs', default=5, help="No. of negative samples")
@click.option('--lam', default=10, help="Dirichlet loss weight")
@click.option('--rho', default=100, help="Classification loss weight")
@click.option('--eta', default=10, help="Diversity loss weight")
@click.option('--window-size', default=10, help="Word embedding context window size")
@click.option('--lr', default=0.01, help="Learning rate")
@click.option('--batch', default=20, help="Batch size")
@click.option('--gpu', default=0, help="CUDA device if CUDA is available")
@click.option('--inductive', default=True, help="Whether to use inductive mode")
@click.option('--dropout', default=0.0, help="dropout rate applied on word/concept embedding")
@click.option('--nepochs', default=10, help="No. of epochs")
@click.option('--concept-metric', default="dot",
              type=click.Choice(['dot', 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
                                 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                                 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                                 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule']),
              help="Distance metric type")
def train(dataset, nconcepts, out_dir, embed_dim, vocab_size, nnegs, lam, rho, eta,
          window_size, lr, batch, gpu, inductive, dropout, nepochs, concept_metric):
    """Train FCM

    DATASET is the name of the dataset to be used. It must be one of the datasets defined in `dataset/`
        which subclass BaseDataset
    OUT-DIR is the path to the output directory where the model, results, and visualization will be saved
    """
    dataset_class = get_dataset(dataset)
    ds = dataset_class()
    data_attr = ds.load_data({"vocab_size": vocab_size, "window_size": window_size})
    fcminer = FocusedConceptMiner(out_dir, embed_dim=embed_dim, nnegs=nnegs, nconcepts=nconcepts,
                                  lam=lam, rho=rho, eta=eta, gpu=gpu, inductive=inductive, **data_attr)
    fcminer.fit(lr=lr, nepochs=nepochs, batch_size=batch)
    fcminer.visualize()


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.argument('config', type=click.Path(exists=True, dir_okay=False))
def grid_search(config):
    """Perform grid search with the given configuration

    CONFIG is the path of the config file
    """
    from grid_search import grid_search as gs
    gs(config)


if __name__ == '__main__':
    fcm()
