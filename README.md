# Pattern Recognition Assignment 2

# Setup

We recommend creating a virtualenv to ensure the dependencies match:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Genes pipeline

The `data.csv` and `labels.csv` files must be located in `Data/genes`.

Run with `python genes.py`. Behaviour can be configured using command line arguments, as follows:

```
usage: genes.py [-h] --reduce_dims {pca,mutual_information} --classifier {svm,naive_bayes,knn} [--grid_search]
```

## Big Cats pipeline

All images must be located in `Data/BigCats`, with subdirectories for each class.

Run with `python Big_Cats_Pipeline.py`. Behaviour can be configured by a few variables in the script.

## Task 2

Data is read from `Data/creditcard.csv`.

Run with `python task2.py`. No configuration is possible.