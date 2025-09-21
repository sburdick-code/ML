# Resources
#  + https://matplotlib.org/
#  + https://numpy.org/doc/

import matplotlib.pyplot as plt
import csv
import const
import numpy as np
from get_scaled_features import Z_normalize_data


def graph_linear_regression(w, b):
    """
    For graphing the training set
    """

    x_train = []
    y_train = []
    X_test = []
    y_pred = []

    x_train_0 = []
    x_train_1 = []
    x_train_2 = []

    x_test_0 = []
    x_test_1 = []
    x_test_2 = []

    # Get Training Data
    with open(const.TRAINING_SET, mode="r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            x_train_0.append(int(row["1stFlrSF"]))
            x_train_1.append(int(row["YearBuilt"]))
            x_train_2.append(int(row["OverallCond"]))
            y_train.append(int(row["SalePrice"]))

    # Get Test Data
    with open(const.TEST_SET, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            x_test_0.append(int(row["1stFlrSF"]))
            x_test_1.append(int(row["YearBuilt"]))
            x_test_2.append(int(row["OverallCond"]))
            X_test.append([x_test_0[-1], x_test_1[-1], x_test_2[-1]])

    X_norm = Z_normalize_data(np.array(X_test))
    f_wb = np.dot(X_norm, w) + b

    fig, axs = plt.subplots(3, 1, layout="constrained")
    axs[0].set_xlabel("1st Floor sqft")
    axs[0].set_ylabel("Sale Price")
    axs[0].plot(x_train_0, y_train, "x", c="blue", label="Training Set")
    axs[0].plot(x_test_0, f_wb, "x", c="red", label="Model Prediction")
    axs[0].legend()

    axs[1].set_xlabel("Year Built")
    axs[1].set_ylabel("Sale Price")
    axs[1].plot(x_train_1, y_train, "x", c="blue", label="Training Set")
    axs[1].plot(x_test_1, f_wb, "x", c="red", label="Model Prediction")
    axs[1].legend()

    axs[2].set_xlabel("Overall Condition")
    axs[2].set_ylabel("Sale Price")
    axs[2].plot(x_train_2, y_train, "x", c="blue", label="Training Set")
    axs[2].plot(x_test_2, f_wb, "x", c="red", label="Model Prediction")
    axs[2].legend()


def graph_J(J_hist):
    fig, ax = plt.subplots()

    ax.set_title("Cost Over Time")
    ax.set_ylabel("cost")
    ax.set_xlabel("iteration")

    ax.plot(J_hist)


def graph_p_hist(p_hist, J_hist):

    w_hist = []
    b_hist = []

    for param in p_hist:
        w_hist.append(param[0])
        b_hist.append(param[1])

    fig, axs = plt.subplots(2, 1, layout="constrained")

    axs[0].set_title("J / w")
    axs[0].plot(w_hist, J_hist, "x", c="green")

    axs[1].set_title("J / b")
    axs[1].plot(b_hist, J_hist, "x", c="green")
