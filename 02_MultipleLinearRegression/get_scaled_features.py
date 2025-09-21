import csv
import numpy as np
import const
import matplotlib.pyplot as plt


def get_scaled_training_data():
    num_x_fts = 3

    X = []
    X_scaled = []
    X_min = np.zeros(num_x_fts)
    X_max = np.zeros(num_x_fts)

    y = []

    # Read data from file
    with open(const.TRAINING_SET, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            x_set = [
                int(row["1stFlrSF"]),
                int(row["YearBuilt"]),
                int(row["OverallCond"]),
            ]

            # Find Min and Max
            for i in range(num_x_fts):
                if x_set[i] > X_max[i]:
                    X_max[i] = x_set[i]

                elif x_set[i] < X_min[i]:
                    X_min[i] = x_set[i]

            # Create an array of unscaled data
            X.append(x_set)
            y.append(int(row["SalePrice"]))

    # Scale X features
    mu = np.mean(X, axis=0)

    for x_set in X:
        x_i_scaled = [0, 0, 0]

        for i, x_i in enumerate(x_set):
            x_i_cur = (x_i - mu[i]) / (X_max[i] - X_min[i])
            x_i_scaled[i] = x_i_cur

        X_scaled.append(x_i_scaled)

    return np.array(X_scaled), np.array(y)


def get_training_data():
    X = []
    y = []
    X_features = ["1stFlrSF", "YearBuilt", "OverallCond"]

    with open(const.TRAINING_SET, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            x_set = [
                int(row[X_features[0]]),
                int(row[X_features[1]]),
                int(row[X_features[2]]),
            ]
            X.append(x_set)
            y.append(int(row["SalePrice"]))

    return np.array(X), np.array(y)


def Z_normalize_data(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    return X_norm


if __name__ == "__main__":
    # Print Training Data - Testing Only
    X_train, y_train = get_training_data()
    X_features = ["1stFlrSF", "YearBuilt", "OverallCond"]
    m, n = X_train.shape

    # Z-Score Normalization
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    X_mean = X_train - mu
    X_norm = (X_train - mu) / sigma

    # Standard Normalization
    X_scaled, y_scaled = get_scaled_training_data()

    # Print a table of scaled data
    print(f"Unnormalized \t :\t\t\t z-Score Norm \t\t\t\t :\t\t\t Standard Norm")
    for i in range(m):
        print(f"{X_train[i]} : {X_norm[i]} : {X_scaled[i]}")

    # Graph
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].scatter(X_train[:, 0], X_train[:, 2])
    ax[0].set_xlabel(X_features[0])
    ax[0].set_ylabel(X_features[2])
    ax[0].set_title("unnormalized")
    ax[0].axis("equal")

    ax[1].scatter(X_mean[:, 0], X_mean[:, 2])
    ax[1].set_xlabel(X_features[0])
    ax[0].set_ylabel(X_features[2])
    ax[1].set_title(r"X - $\mu$")
    ax[1].axis("equal")

    ax[2].scatter(X_norm[:, 0], X_norm[:, 2])
    ax[2].set_xlabel(X_features[0])
    ax[0].set_ylabel(X_features[2])
    ax[2].set_title(r"Z-score normalized")
    ax[2].axis("equal")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("distribution of features before, during, after normalization")
    plt.show()
