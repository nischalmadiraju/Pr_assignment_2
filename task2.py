from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn import model_selection, semi_supervised, metrics, linear_model


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv('creditcard.csv')

    labels = data['Class']
    values = data[[f'V{i + 1}' for i in range(28)]]

    return values.to_numpy(), labels.to_numpy()


def evaluate_model(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> \
        Tuple[float, float]:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    # print(metrics.confusion_matrix(y_test, y_pred))
    print(f'{f1=:.3f}\t{accuracy=:.3f}')
    return accuracy, f1


def main():
    print('Loading data')
    data, labels = load_data()

    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=.2, stratify=labels)
    x_train_lab, x_train_unlab, y_train_lab, _ = model_selection.train_test_split(x_train, y_train, test_size=.7,
                                                                                  stratify=y_train)

    print('Baseline model')
    baseline = linear_model.LogisticRegression()
    evaluate_model(baseline, x_train_lab, y_train_lab, x_test, y_test)

    # Same as previous x_train, but shuffled by train_test_split
    x_train = np.concatenate((x_train_lab, x_train_unlab))
    y_train_combined = np.concatenate((y_train_lab, np.full((x_train_unlab.shape[0],), -1)))

    print('Semi-supervised')
    semi_model = semi_supervised.LabelPropagation(kernel='knn', n_jobs=-1)
    evaluate_model(semi_model, x_train, y_train_combined, x_test, y_test)

    # TODO is this correct?
    y_train = semi_model.transduction_

    print('Baseline with complete dataset')
    baseline_complete = linear_model.LogisticRegression()
    evaluate_model(baseline_complete, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
