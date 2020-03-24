# Focused Concept Miner 

End-to-end deep learning framework for automatic mining of "focused" concepts.

## Prerequisites

   * Get the code: `git clone git@github.com:ecfm/fcm_cli.git; cd fcm_cli`
   * Install and activate the environment:
```bash
python3 -m venv env
source env/bin/activate
cd src
pip install --editable .
```

## Quickstart

   * View help info `fcm --help`
   * Train with 20 News Group dataset with default parameters and save the model, results, and visualization 
   html in the folder "run0": ```fcm train news_group run0```
   * Start grid search for 20 News Group dataset with an example hyperparameter search space configuration: ```fcm grid-search ../configs/news_config.json```

## Usage
   * Usage of `fcm train`:
   ```text
Usage: fcm train [OPTIONS] DATASET OUT_DIR

  DATASET is the name of the dataset to be used. It must be one of the
  datasets defined in `dataset/`     which subclass BaseDataset OUT-DIR is
  the path to the output directory where the model, results, and
  visualization will be saved

Options:
  --nconcepts INTEGER             No. of concepts
  --embed-dim INTEGER             The size of each word/concept embedding
                                  vector
  --vocab-size INTEGER            Maximum vocabulary size
  --nnegs INTEGER                 No. of negative samples
  --lam INTEGER                   Dirichlet loss weight
  --rho INTEGER                   Classification loss weight
  --eta INTEGER                   Diversity loss weight
  --window-size INTEGER           Word embedding context window size
  --lr FLOAT                      Learning rate
  --batch INTEGER                 Batch size
  --gpu INTEGER                   CUDA device if CUDA is available
  --inductive TEXT                Whether to use inductive mode
  --dropout FLOAT                 dropout rate applied on word/concept
                                  embedding

  --nepochs INTEGER               No. of epochs
  --concept-metric [dot|braycurtis|canberra|chebyshev|cityblock|correlation|cosine|dice|euclidean|hamming|jaccard|kulsinski|mahalanobis|matching|minkowski|rogerstanimoto|russellrao|seuclidean|sokalmichener|sokalsneath|sqeuclidean|wminkowski|yule]
                                  Distance metric type
  -h, --help                      Show this message and exit.
```
   * The following is a commented version of the example configuration `fcm_cli/configs/news_config.json`:
   
   ```json
{
  "dataset": "news_group",
  "mode": "product",
  "gpus": [0],
  "max_threads": 1,
  "out_dir": "../grid_search/news/run0",
  "dataset_params": {
    "window_size": [4],
    "min_df": [0.01],
    "max_df": [0.8]
  },
  "fcm_params": {
    "inductive": [true],
    "inductive_dropout": [0.01],
    "embed_dim": [50],
    "hidden_size": [100],
    "num_layers": [1],
    "nnegs": [15],
    "nconcepts": [2],
    "lam": [10],
    "rho": [1000],
    "eta": [1000]
  },
  "fit_params": {
    "lr": [0.01],
    "nepochs": [30],
    "pred_only_epochs": [15],
    "batch_size": [10240],
    "grad_clip": [1024]
  }
}
```
