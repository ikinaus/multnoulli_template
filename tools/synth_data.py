from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_data(n_samples=1000, n_features=5, n_classes=3):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_classes=n_classes,
        random_state=42,
    )

    Y_one_hot = np.eye(n_classes)[y]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_one_hot, test_size=0.2, random_state=42
    )

    train_data = np.hstack((X_train, Y_train))
    test_data = np.hstack((X_test, Y_test))

    return train_data, test_data


def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir.parent / "data" / "processed"

    train_data, test_data = get_data(n_samples=5000)
    cols = "x1,x2,x3,x4,x5,y1,y2,y3"

    save_configs = [("ds_train.csv", train_data), ("ds_test.csv", test_data)]

    for filename, data in save_configs:
        file_path = output_dir / filename
        np.savetxt(file_path, data, delimiter=",", header=cols, comments="", fmt="%.6f")


if __name__ == "__main__":
    main()
