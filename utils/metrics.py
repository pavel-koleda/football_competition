from typing import Union

import numpy as np


def accuracy_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Accuracy score.

    The formula is as follows:
        accuracy = (1 / N) Σ(i=0 to N-1) I(y_i == t_i),

        where:
            - N - number of samples,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i) - indicator function.
    Args:
        targets: The true labels.
        predictions: The predicted classes.
    """
    # TODO: Implement calculation of accuracy
    raise NotImplementedError


def accuracy_score_per_class(targets: np.ndarray, predictions: np.ndarray) -> list[float]:
    """Accuracy score for each class.

    The formula is as follows:
        accuracy_k = (1 / N_k) Σ(i=0 to N) I(y_i == t_i) * I(t_i == k)

        where:
            - N_k -  number of k-class elements,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i), I(t_i == k) - indicator function.

    Args:
        targets: The true labels.
        predictions: The predicted classes.

    Returns:
        list[float]: Accuracy for each class.
    """
    accuracy_per_class = []
    # TODO: Implement calculation of accuracy for each class using the formula from the docstring
    raise NotImplementedError


def balanced_accuracy_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Balanced accuracy score.

    The formula is as follows:
        balanced_accuracy = (1 / K) Σ(k=0 to K-1) accuracy_k,
        accuracy_k = (1 / N_k) Σ(i=0 to N) I(y_i == t_i) * I(t_i == k)

        where:
            - K - number of classes,
            - N_k - number of k-class elements,
            - accuracy_k - accuracy for k-class,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i), I(t_i == k) - indicator function.

    Args:
        targets: The true labels.
        predictions: The predicted classes.
    """
    # TODO: Implement calculation of balanced accuracy using the formula from the docstring
    raise NotImplementedError


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, classes_num: Union[int, None] = None) -> np.ndarray:
    """Confusion matrix.

    Confusion matrix C with shape KxK:
        c[i, j] - number of observations known to be in class i and predicted to be in class j,

        where:
            - K is the number of classes.

    Args:
        targets: The true labels.
        predictions: The predicted classes.
        classes_num: The number of unique classes.
    """
    # TODO: Implement calculation of confusion matrix using the formula from the docstring
    raise NotImplementedError
