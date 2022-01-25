from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import model_selection, semi_supervised, metrics


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv('creditcard.csv')

    labels = data['Class']
    values = data[[f'V{i + 1}' for i in range(28)]]

    return values.to_numpy(), labels.to_numpy()


def main():
    print('Loading data')
    data, labels = load_data()

    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=.2, stratify=labels)
    x_train_lab, x_train_unlab, y_train_lab, _ = model_selection.train_test_split(x_train, y_train, test_size=.7,
                                                                                  stratify=y_train)

    # TODO train baseline model here with only x_train_lab

    # Same as previous x_train, but shuffled by train_test_split
    x_train = np.concatenate((x_train_lab, x_train_unlab))
    y_train_combined = np.concatenate((y_train_lab, np.full((x_train_unlab.shape[0],), -1)))

    print('Propagating labels')
    model = semi_supervised.LabelPropagation(kernel='knn', n_jobs=-1)
    model.fit(x_train, y_train_combined)
    semi_f1 = metrics.f1_score(y_test, model.predict(x_test))
    print('Semi-supervised F1', semi_f1)

    # TODO is this correct?
    y_train = np.concatenate((y_train_lab, model.transduction_))

    # TODO retrain baseline with new y_train


if __name__ == '__main__':
    main()
