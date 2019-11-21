# Focused Concept Miner 

End-to-end deep learning framework for automatic mining of "focused" concepts.

## Prerequisites

   * Get the code: `git clone git@github.com:ecfm/fcm_cli.git; cd fcm_cli`
   * Install and activate the environment:
```
python3 -m venv env
source env/bin/activate
cd src
pip install --editable .
```

## Quickstart

   * View help info `fcm --help`
   * Training with 20 News Group dataset with default parameters and save the model, results, and visualization 
   html in the folder "run0": ```fcm train news_group run0``` 
