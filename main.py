import argparse
from pprint import pprint
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition, feature_selection, preprocessing, svm, naive_bayes, cluster, metrics, model_selection
from sklearn.metrics import make_scorer


def pca(data: np.ndarray, *_) -> np.ndarray:
    return decomposition.PCA(n_components=0.95).fit_transform(data)


def mutual_information(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mi_scores = feature_selection.mutual_info_classif(x, y)
    mi_score_selected_index = np.where(mi_scores > 0.2)[0]
    return x[:, mi_score_selected_index]


reduce_dims_options = {
    'pca': pca,
    'mutual_information': mutual_information,
}


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return metrics.f1_score(y_true, y_pred, average='weighted')


classifier_options = {
    'svm': svm.SVC,
    'naive_bayes': naive_bayes.GaussianNB,
}

grid_options = {
    'svm': {
        'C': [.2, .5, 1.],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    },
    'naive_bayes': {
        'var_smoothing': [1e-7, 1e-9, 1e-11]
    },
}


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv('data.csv', index_col=0)
    labels = pd.read_csv('labels.csv', index_col=0)

    return data.to_numpy(), labels.to_numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduce_dims', choices=reduce_dims_options.keys(), required=True)
    parser.add_argument('--classifier', choices=classifier_options.keys(), required=True)
    parser.add_argument('--grid_search', action='store_true')

    return parser.parse_args()


def kmeans(x: np.ndarray, y: np.ndarray) -> float:
    kmeans = cluster.KMeans(n_clusters=2, random_state=0)
    y_pred = kmeans.fit_predict(x)
    # Put the result into a color plot
    plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.show()
    return metrics.mutual_info_score(y, y_pred)


def main():
    args = parse_args()
    print('Loading data')
    data, labels = load_data()

    label_encoder = preprocessing.OrdinalEncoder()
    enc_labels = label_encoder.fit_transform(labels)
    enc_labels = np.ravel(enc_labels)

    print('Original clusters MI', kmeans(data, enc_labels))

    print('Reducing dims using', args.reduce_dims)
    reduced = reduce_dims_options[args.reduce_dims](data, enc_labels)

    print('Reduced clusters MI', kmeans(reduced, enc_labels))

    x_train, x_test, y_train, y_test = model_selection.train_test_split(reduced, enc_labels, test_size=.2)

    print('Training classifier', args.classifier)
    model = classifier_options[args.classifier]()

    if args.grid_search:
        print('Performing grid search')
        params = grid_options[args.classifier]

        # Automatically uses KFold CV
        grid = model_selection.GridSearchCV(model, params, scoring=make_scorer(compute_f1))
        model = grid.fit(x_train, y_train)
        print('Best parameters')
        pprint(grid.best_params_)
    else:
        model = model.fit(x_train, y_train)

    print('Test F1', compute_f1(y_test, model.predict(x_test)))


if __name__ == '__main__':
    main()
