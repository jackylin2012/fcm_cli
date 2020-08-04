#!/usr/bin/env python
import os
import sys

import click

from concept_viewer_app.make_fixtures import save_fixtures
from dataset.csv import CSVDataset

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
@click.option('--csv-path', type=click.Path(dir_okay=False), help="Path to the csv file. "
                                                                  "Only needed if `dataset` is 'csv'")
@click.option('--csv-text', default="", help="Column name of the text field in the csv file. "
                                             "Only needed if `dataset` is 'csv'")
@click.option('--csv-label', default="", help="Column name of the label field in the csv file. "
                                              "Only needed if `dataset` is 'csv'")
@click.option('--nconcepts', default=5, help="No. of concepts (default: 5)")
@click.option('--embed-dim', default=50, help="The size of each word/concept embedding vector (default: 50)")
@click.option('--vocab-size', default=5000, help="Maximum vocabulary size (default: 5000)")
@click.option('--nnegs', default=5, help="No. of negative samples (default: 5)")
@click.option('--lam', default=10, help="Dirichlet loss weight (default: 10)")
@click.option('--rho', default=100, help="Prediction loss weight (default: 100)")
@click.option('--eta', default=10, help="Diversity loss weight (default: 10)")
@click.option('--window-size', default=4, help="Word embedding context window size (default: 4)")
@click.option('--lr', default=0.01, help="Learning rate (default: 0.01)")
@click.option('--batch', default=20, help="Batch size (default: 20)")
@click.option('--gpu', default=0, help="GPU device if CUDA is available. Ignored if no CUDA. (default: 0)")
@click.option('--inductive', default=True, help="Whether to use inductive mode (default: True)")
@click.option('--dropout', default=0.0, help="dropout rate applied on word/concept embedding (default: 0.0)")
@click.option('--nepochs', default=10, help="No. of epochs (default: 10)")
@click.option('--concept-dist', default="dot",
              type=click.Choice(['dot', 'correlation', 'cosine', 'euclidean', 'hamming']),
              help="Concept vectors distance metric (default: 'dot')")
def train(dataset, csv_path, csv_text, csv_label, nconcepts, out_dir, embed_dim, vocab_size, nnegs, lam, rho, eta,
          window_size, lr, batch, gpu, inductive, dropout, nepochs, concept_dist):
    """Train FCM

    DATASET is the name of the dataset to be used. It must be one of the datasets defined in `fcm_cli/src/dataset/`
    which is a subclass of BaseDataset.

    OUT-DIR is the path to the output directory where the model, results, and visualization will be saved
    """
    if dataset == "csv":
        ds = CSVDataset(csv_path, csv_text, csv_label)
    else:
        dataset_class = get_dataset(dataset)
        ds = dataset_class()
    print("Loading data...")
    data_attr = ds.load_data({"vocab_size": vocab_size, "window_size": window_size})
    # remove gensim keys which are only used for visualization
    del data_attr["gensim_corpus"]
    del data_attr["gensim_dictionary"]
    fcminer = FocusedConceptMiner(out_dir, embed_dim=embed_dim, nnegs=nnegs, nconcepts=nconcepts,
                                  lam=lam, rho=rho, eta=eta, gpu=gpu, file_log=True, inductive=inductive, **data_attr)
    print("Starts training")
    fcminer.fit(lr=lr, nepochs=nepochs, batch_size=batch, concept_dist=concept_dist)
    fcminer.visualize()
    print("Training finished. See results in " + out_dir)


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.argument('config', type=click.Path(exists=True, dir_okay=False))
def grid_search(config):
    """Perform grid search with the given configuration

    CONFIG is the path of the config file
    """
    from grid_search import grid_search as gs
    gs(config)


@fcm.command(context_settings=CONTEXT_SETTINGS)
@click.argument('dir', type=click.Path(exists=True, file_okay=False))
def visualize(dir):
    """Visualize the concept words in a grid search result directory

    DIR is the path to the grid search result directory
    """
    from concept_viewer_app.manage import execute
    execute(["concept_viewer_app/manage.py", "makemigrations"])
    execute(["concept_viewer_app/manage.py", "migrate", "--run-syncdb"])
    execute(["concept_viewer_app/manage.py", "flush", "--no-input"])
    saved_file = save_fixtures(dir)
    execute(["concept_viewer_app/manage.py", "loaddata", saved_file])
    execute(["concept_viewer_app/manage.py", "runserver", "8000"])


if __name__ == '__main__':
    fcm()
