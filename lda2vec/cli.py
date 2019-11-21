#!/usr/bin/env python
import importlib
import os
import sys

import click

from SLda2vec import SLda2vec
from dataset.base_dataset import BaseDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group()
@click.pass_context
def fcm(ctx):
    pass


def get_dataset(dataset):
    module_path = "dataset.%s" % dataset
    module = importlib.import_module(module_path)
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if issubclass(obj, BaseDataset) and obj != BaseDataset:
                return obj
        except TypeError:  # If 'obj' is not a class
            pass
    raise ImportError("Cannot find a subclass of %s in %s" % (BaseDataset, module_path))


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.option('--dataset', help="Name of the dataset to use")
@click.option('--ntopics', default=5, help="No. of topics")
@click.option('--out-dir', default='run0', help="Output model folder")
@click.option('--embed-size', default=50, help="Word/topic embedding size")
@click.option('--vocab-size', default=10000, help="Maximum vocabulary size")
@click.option('--pretrained-embed', type=str, default=None, help="Pretrained word embedding path")
@click.option('--nnegs', default=5, help="No. of negative samples")
@click.option('--lam', default=10, help="Dirichlet loss weight")
@click.option('--rho', default=100, help="Classification loss weight")
@click.option('--eta', default=10, help="Diversity loss weight")
@click.option('--window-size', default=10, help="Word embedding context window size")
@click.option('--lr', default=0.001, help="Learning rate")
@click.option('--batch', default=20, help="Batch size")
@click.option('--device', default=0, help="CUDA device if CUDA is available")
@click.option('--dropout', default=0.0, help="dropout rate applied on word/topic embedding")
@click.option('--nepochs', default=10, help="No. of epochs")
@click.pass_context
def train(ctx, dataset, ntopics, out_dir, embed_size, vocab_size, pretrained_embed, nnegs, lam, rho, eta,
          window_size, lr, batch, device, dropout, nepochs):
    """Train FCM"""
    dataset_class = get_dataset(dataset)
    ds = dataset_class()
    data_attr = ds.load_data(vocab_size=vocab_size, window_size=window_size)
    l2v = SLda2vec(embed_size=embed_size, nepochs=nepochs, nnegs=nnegs,
                   ntopics=ntopics, lam=lam, rho=rho, eta=eta, **data_attr)
    l2v.fit(batch_size=batch, lr=lr)
    l2v.visualize("run0")


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.argument('config')
@click.pass_context
def grid_search(ctx, config):
    """Perform grid search with the given configuration

    CONFIG is the path of the config file
    """
    grid_search(config)


if __name__ == '__main__':
    fcm(obj={})
