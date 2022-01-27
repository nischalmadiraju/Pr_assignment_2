from typing import Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import model_selection, semi_supervised, metrics, svm

np.random.seed(42)


def transform_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Randomly take n rows with class 0, where n is the count of class 1 items.
    class0_equal = data[data['Class'] == 0].sample(data['Class'].value_counts()[1])
    equal = pd.concat((data[data['Class'] == 1], class0_equal))

    labels = equal['Class']
    values = equal[[f'V{i + 1}' for i in range(28)]]

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


def run_experiment(data: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    data, labels = transform_data(data)

    assert (labels == 0).sum() == (labels == 1).sum()

    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=.2, stratify=labels)
    x_train_lab, x_train_unlab, y_train_lab, _ = model_selection.train_test_split(x_train, y_train, test_size=.7,
                                                                                  stratify=y_train)

    print('Baseline model')
    baseline = svm.SVC()
    baseline_result = evaluate_model(baseline, x_train_lab, y_train_lab, x_test, y_test)

    # Same as previous x_train, but shuffled by train_test_split
    x_train = np.concatenate((x_train_lab, x_train_unlab))
    y_train_combined = np.concatenate((y_train_lab, np.full((x_train_unlab.shape[0],), -1)))

    print('Semi-supervised')
    semi_model = semi_supervised.LabelPropagation(kernel='knn', n_jobs=-1)
    semi_result = evaluate_model(semi_model, x_train, y_train_combined, x_test, y_test)

    # TODO is this correct?
    y_train = semi_model.transduction_

    print('Baseline with complete dataset')
    baseline_complete = svm.SVC()
    baseline_complete_result = evaluate_model(baseline_complete, x_train, y_train, x_test, y_test)

    return baseline_result, semi_result, baseline_complete_result


def main():
    print('Loading data')
    data = pd.read_csv('Data/creditcard.csv')

    print(data['Class'].value_counts())

    results = []
    for i in range(1_000):
        print(f'Running experiment {i}')
        results.append(run_experiment(data))
        print('\n' * 2)

    results = np.array(results)
    f1 = results[:, :, 0]
    acc = results[:, :, 1]

    plt.title('F1-score')
    plt.boxplot(f1)
    plt.xticks([1, 2, 3], ['Baseline', 'Semi-supervised', 'Complete'])
    plt.show()

    plt.title('Accuracy')
    plt.boxplot(acc)
    plt.xticks([1, 2, 3], ['Baseline', 'Semi-supervised', 'Complete'])
    plt.show()

    print('F1', stats.ttest_ind(f1[:, 0], f1[:, 2]))
    print('Accuracy', stats.ttest_ind(acc[:, 0], acc[:, 2]))


if __name__ == '__main__':
    main()
