import os

import numpy as np
from sklearn.metrics import accuracy_score

from src.config import PROCESSED_DATA_PATH
from src.model import Multinoulli
from src.utils import add_intercept_column

TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "ds_train.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "ds_test.csv")


def load_and_preprocess_data(file_name, add_intercept=True):
    with open(file_name, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    if header_line.startswith("#"):
        header_line = header_line[1:].strip()

    columns = header_line.split(",")

    y_count = sum(1 for col in columns if col.strip().lower().startswith("y"))

    if y_count == 0:
        raise ValueError(
            f"No target columns starting with 'y' found in header of {file_name}"
        )

    # 2. Load the numerical data
    data = np.loadtxt(file_name, delimiter=",", skiprows=1)

    X = data[:, :-y_count]
    Y = data[:, -y_count:]

    if add_intercept:
        X = add_intercept_column(X)

    return X, Y


def main():
    X_train, Y_train = load_and_preprocess_data(TRAIN_DATA_PATH)
    X_test, Y_test = load_and_preprocess_data(TEST_DATA_PATH)

    classifier = Multinoulli(verbose=True)
    classifier.fit(X_train, Y_train)

    predicted_labels = classifier.predict(X_test)

    accuracy = accuracy_score(np.argmax(Y_test, axis=1), predicted_labels)

    print(f"Accuracy score: {accuracy:.4f}")
    print(f"Weights: {classifier.get_weights()}")


if __name__ == "__main__":
    main()
