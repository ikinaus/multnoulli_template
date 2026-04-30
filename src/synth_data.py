import numpy as np
from sklearn.datasets import make_classification

from src.utils import add_intercept_column


def get_test_data(n_samples=1000, n_features=5, n_classes=3):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_classes=n_classes,
        random_state=42,
    )

    X = add_intercept_column(X)

    Y_one_hot = np.eye(n_classes)[y]

    return X, Y_one_hot
