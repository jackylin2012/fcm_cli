#!/usr/bin/env python

import os
import sys

import click

from SLda2vec import SLda2vec

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group()
@click.option('--ntopics', default=5, help="No. of topics")
@click.option('--out-dir', default='run0', help="Output model folder")
@click.option('--vocab-size', default=20000, help="Vocabulary size")
@click.option('--embedding-size', default=50, help="Word/topic embedding size")
@click.option('--pretrained-embed', type=str, default=None, help="Pretrained word embedding path")
@click.option('--nnegs', default=5, help="No. of negative samples")
@click.option('--lam', default=10, help="Dirichlet loss weight")
@click.option('--rho', default=100, help="Classification loss weight")
@click.option('--eta', default=10, help="Diversity loss weight")
@click.option('--nreviews', default=2, help="No. of reviews")
@click.pass_context
def fcm(ctx, ntopics, out_dir, vocab_size, embedding_size, pretrained_embed, nnegs, lam, rho, eta, nreviews):
    ctx.ensure_object(dict)
    ctx.obj['l2v'] = SLda2vec(vocab_size=vocab_size, embedding_size=embedding_size, nepochs=nepochs, nnegs=nnegs,
                              word_counts=wordcounts, ntopics=ntopics, ndocs=ndocs, lam=lam, rho=rho, eta=eta,
                              doc_weights=docweights, doc_topic_weights=doc_topic_weights, word_vectors=word_vectors,
                              theta=linear_weights, gpu=GPU)

    pass


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.option('--dataset', default='wcai', help="Name of the dataset to use")
@click.option('--window-size', default=10, help="Word embedding context window size")
@click.pass_context
def prep(ctx, dataset, window_size):
    """Preprocess documents"""
    pass


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.option('--lr', default=0.001, help="Learning rate")
@click.option('--batch', default=1, help="Batch size")
@click.option('--device', default=0, help="CUDA device if CUDA is available")
@click.option('--dropout', default=0.0, help="dropout rate applied on word/topic embedding")
@click.option('--nepochs', default=100, help="No. of epochs")
@click.pass_context
def train(ctx, lr, batch, device, dropout, nepochs):
    """Train FCM"""

    pass


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.argument('config')
@click.pass_context
def grid_search(ctx, config):
    """Perform grid search with the given configuration

    CONFIG is the path of the config file
    """
    pass


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.argument('output')
@click.pass_context
def visualize(ctx, output):
    """Visualize the training losses and save them to the output folder

    OUTPUT is the path of the output folder
    """
    pass


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.option('--batch', default=1, help="Batch size")
@click.option('--lam', default=10, help="Dirichlet loss weight")
@click.option('--rho', default=100, help="Classification loss weight")
@click.option('--eta', default=10, help="Diversity loss weight")
@click.option('--device', default=0, help="CUDA device if CUDA is available")
@click.pass_context
def test(ctx, batch, lam, rho, eta, device):
    """Test FCM"""
    if ctx.obj['slda'] is None:
        print("'fcm train' should be run first before 'fcm test'")
        return
    print("test")
    print("ctx.obj['slda']="+ctx.obj['slda'])


if __name__ == '__main__':
    fcm(obj={})