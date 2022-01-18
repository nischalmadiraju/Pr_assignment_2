from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv('data.csv', index_col=0)
    labels = pd.read_csv('labels.csv', index_col=0)

    return data, labels


def reduce_dims(data: pd.DataFrame, labels: pd.DataFrame) -> np.ndarray:

    X = data.to_numpy()
    y = labels.to_numpy()
    mi_scores = mutual_info_classif(X, np.ravel(y))
    mi_score_selected_index = np.where(mi_scores > 0.2)[0]
    reduced = X[:,mi_score_selected_index]
    return reduced


def train_classifier(data: np.ndarray, labels: np.ndarray) -> None:
    label_encoder = preprocessing.OrdinalEncoder()
    enc_labels = label_encoder.fit_transform(labels)
    gnb = GaussianNB()
    gnb.fit(data, np.ravel(enc_labels))

    train_acc = gnb.score(data, enc_labels)
    print(f'Accuracy on training data: {train_acc}')


def main():
    print('Loading data')
    data, labels = load_data()
    print('Reducing dims')
    reduced = reduce_dims(data, labels)
    print('Training classifier')
    train_classifier(reduced, labels)


if __name__ == '__main__':
    main()
