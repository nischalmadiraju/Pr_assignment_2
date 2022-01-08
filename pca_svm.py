from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing, svm, decomposition


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv('data.csv', index_col=0)
    labels = pd.read_csv('labels.csv', index_col=0)

    return data, labels


def reduce_dims(data: pd.DataFrame) -> np.ndarray:
    pca = decomposition.PCA(n_components=0.95)
    np_data = data.to_numpy()
    reduced = pca.fit_transform(np_data)
    return reduced


def train_classifier(data: np.ndarray, labels: np.ndarray) -> None:
    label_encoder = preprocessing.OrdinalEncoder()
    enc_labels = label_encoder.fit_transform(labels)

    svc = svm.SVC()
    svc.fit(data, np.ravel(enc_labels))

    train_acc = svc.score(data, enc_labels)
    print(f'Accuracy on training data: {train_acc}')


def main():
    print('Loading data')
    data, labels = load_data()
    print('Reducing dims')
    reduced = reduce_dims(data)
    print('Training classifier')
    train_classifier(reduced, labels.to_numpy())


if __name__ == '__main__':
    main()
