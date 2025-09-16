import csv
import numpy as np
import const


def get_training_data():

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
    for x_set in X:
        x_i_scaled = [0, 0, 0]

        for i, x_i in enumerate(x_set):
            x_i_cur = (x_i - X_min[i]) / (X_max[i] - X_min[i])
            x_i_scaled[i] = x_i_cur

        X_scaled.append(x_i_scaled)

    return np.array(X_scaled), np.array(y)


if __name__ == "__main__":
    X_train, y_train = get_training_data()
    m, n = X_train.shape

    for x in X_train:
        print(x)
