import argparse
from pprint import pprint
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition, feature_selection, preprocessing, svm, naive_bayes, cluster, metrics, \
    model_selection, neighbors
from sklearn.metrics import make_scorer

# Set seed for reproducible modeling
np.random.seed(42)


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
    'knn': neighbors.KNeighborsClassifier,
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
    data = pd.read_csv('Data/genes/data.csv', index_col=0)
    labels = pd.read_csv('Data/genes/labels.csv', index_col=0)

    print('Data shape', data.shape)

    labels['Class'].value_counts().plot(title='Items per class', rot=0, kind='bar')
    plt.show()

    return data.to_numpy(), labels.to_numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduce_dims', choices=reduce_dims_options.keys(), required=True)
    parser.add_argument('--classifier', choices=classifier_options.keys(), required=True)
    parser.add_argument('--grid_search', action='store_true')

    return parser.parse_args()


def kmeans(x: np.ndarray, y: np.ndarray) -> float:
    unique_clusters = np.unique(y)
    clusters = cluster.KMeans(n_clusters=len(unique_clusters), random_state=0)
    y_pred = clusters.fit_predict(x)
    # Put the result into a color plot
    for i, name in enumerate(unique_clusters):
        plt.scatter(x[y_pred == i, i], x[y_pred == i, i + 1], s=100, label=name)
    plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], s=300, c='gray', label='Centroids')
    plt.show()
    return metrics.mutual_info_score(y, y_pred)


def main():
    args = parse_args()

    if args.classifier == 'knn' and args.grid_search:
        raise ValueError('Grid search can not be performed for knn')

    print('Loading data')
    data, labels = load_data()

    label_encoder = preprocessing.OrdinalEncoder()
    enc_labels = label_encoder.fit_transform(labels)
    enc_labels = np.ravel(enc_labels)

    print(f'Original clusters MI = {kmeans(data, enc_labels):.3f}')

    print('Reducing dims using', args.reduce_dims)
    reduced = reduce_dims_options[args.reduce_dims](data, enc_labels)

    print(f'Reduced clusters MI = {kmeans(reduced, enc_labels):.3f}')

    x_train, x_test, y_train, y_test = model_selection.train_test_split(reduced, enc_labels, test_size=.2)

    print('Training classifier', args.classifier)

    if args.grid_search:
        model = classifier_options[args.classifier]()
        print('Performing grid search')
        params = grid_options[args.classifier]

        # Automatically uses KFold CV
        grid = model_selection.GridSearchCV(model, params, scoring=make_scorer(compute_f1))
        final_model = grid.fit(x_train, y_train)
        print('Best parameters')
        pprint(grid.best_params_)
    else:
        kfold_results = []
        for train_idx, val_idx in model_selection.StratifiedKFold().split(x_train, y_train):
            model = classifier_options[args.classifier]()
            model.fit(x_train[train_idx], y_train[train_idx])

            val_f1 = compute_f1(y_train[val_idx], model.predict(x_train[val_idx]))
            print(f'KFold F1 = {val_f1:.3f}')
            kfold_results.append((val_f1, model))

        f1_scores = [f1 for f1, _ in kfold_results]
        print(f'KFold F1 statistics = {np.mean(f1_scores):.3f} \pm {np.std(f1_scores):.3f}')

        final_model = classifier_options[args.classifier]()
        final_model.fit(x_train, y_train)

    print(f'Test F1 = {compute_f1(y_test, final_model.predict(x_test)):.3f}')


if __name__ == '__main__':
    main()
