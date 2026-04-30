import os

import numpy as np
from sklearn.metrics import accuracy_score

from src.config import RAW_DATA_PATH
from src.model import LogReg
from src.utils import add_intercept_column

TRAIN_DATA_PATH = os.path.join(RAW_DATA_PATH, "ds1_train.csv")
TEST_DATA_PATH = os.path.join(RAW_DATA_PATH, "ds1_test.csv")


def load_and_preprocess_data(file_name, add_intercept=True):
    data = np.loadtxt(file_name, delimiter=",", skiprows=1)

    X = data[:, :-1]
    y = data[:, -1]

    if add_intercept:
        X = add_intercept_column(X)

    return X, y


def main():
    X_train, y_train = load_and_preprocess_data(TRAIN_DATA_PATH)
    X_test, y_test = load_and_preprocess_data(TEST_DATA_PATH)

    classifier = LogReg(lr=(0.01 / X_train.shape[0]), verbose=True)
    classifier.fit(X_train, y_train)

    # predict_prb = classifier.predict_proba(X_test)
    predicted_labels = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predicted_labels)

    print(f"Accuracy score: {accuracy:.4f}")
    print(f"Weights: {classifier.get_weights()}")


if __name__ == "__main__":
    main()
