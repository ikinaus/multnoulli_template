import os

import numpy as np
from sklearn.metrics import accuracy_score

from src.config import PROCESSED_DATA_PATH
from src.model import Multinoulli
from src.utils import add_intercept_column

TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "ds_train.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "ds_test.csv")


def load_and_preprocess_data(file_name, add_intercept=True):
    data = np.loadtxt(file_name, delimiter=",", skiprows=1)

    X = data[:, :-3]
    Y = data[:, -3:]

    if add_intercept:
        X = add_intercept_column(X)

    return X, Y


def main():
    X_train, Y_train = load_and_preprocess_data(TRAIN_DATA_PATH)
    X_test, Y_test = load_and_preprocess_data(TEST_DATA_PATH)

    classifier = Multinoulli(verbose=True)
    classifier.fit(X_train, Y_train)

    # predict_prb = classifier.predict_proba(X_test)
    predicted_labels = classifier.predict(X_test)

    accuracy = accuracy_score(np.argmax(Y_test, axis=1), predicted_labels)

    print(f"Accuracy score: {accuracy:.4f}")
    print(f"Weights: {classifier.get_weights()}")


if __name__ == "__main__":
    main()
