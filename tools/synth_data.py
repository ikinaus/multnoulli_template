from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_data(n_samples=1000, n_features=6, n_classes=5):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_classes=n_classes,
        random_state=42,
    )

    Y_one_hot = np.eye(n_classes)[y]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_one_hot, test_size=0.2, random_state=42
    )

    train_data = np.hstack((X_train, Y_train))
    test_data = np.hstack((X_test, Y_test))

    return train_data, test_data, n_features, n_classes


def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir.parent / "data" / "processed"

    train_data, test_data, nx, ny = get_data(n_samples=10000)
    x_cols = [f"x{col}" for col in range(1, nx + 1)]
    y_cols = [f"y{col}" for col in range(1, ny + 1)]
    cols = ",".join((x_cols + y_cols))

    save_configs = [("ds_train.csv", train_data), ("ds_test.csv", test_data)]

    for filename, data in save_configs:
        file_path = output_dir / filename
        np.savetxt(file_path, data, delimiter=",", header=cols, comments="", fmt="%.6f")


if __name__ == "__main__":
    main()
