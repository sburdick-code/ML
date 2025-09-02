# Resources
#  + https://matplotlib.org/
#  + https://numpy.org/doc/

import matplotlib.pyplot as plt
import numpy as np
import csv
import const


def graph_linear_regression(w, b):
    """
    For graphing the training set
    """

    x_train = []
    y_train = []
    x_test = []
    y_pred = []

    # Get Training Data
    with open(const.training_set, mode="r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            x_train.append(int(row["1stFlrSF"]))
            y_train.append(int(row["SalePrice"]))

    # Get Test Data
    with open(const.test_set, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            x_i = int(row["1stFlrSF"])
            x_test.append(x_i)
            f_wb = w * x_i + b
            y_pred.append(f_wb)

    fig, ax = plt.subplots()
    ax.set_xlabel("1st Floor SqFt")
    ax.set_ylabel("Sale Price")

    ax.plot(x_train, y_train, "x", c="blue", label="Training Set")
    ax.plot(x_test, y_pred, c="red", label="Model Prediction")

    ax.legend()
    plt.show()


# Testing purposes only
def test_graphs():
    fig, ax = plt.subplots()

    ax.set_ylabel("Y-Axis")
    ax.set_xlabel("X-Axis")
    ax.set_title("My Title")

    # GRAPH LINE
    # line1 = ax.plot([1, 2, 3, 4], [1, 2, 3, 4], label="Line 1")

    # GRAPH SCATTER
    # data = {
    #     "a": [1, 2, 3, 4],
    #     "b": [5, 3, 2, 1],
    #     "x": [5, 4, 3, 1],
    #     "y": [1, 2, 3, 4],
    # }
    #
    # values1 = ax.scatter("a", "b", c="green", s=20, data=data)
    # values1.set_label("Group 1")
    # values2 = ax.scatter("x", "y", c="red", s=5, data=data, )
    # values2.set_label("Group 2")

    # GRAPH POINTS
    data1 = [2, 4, 6, 8]
    data2 = [5, 3, 2, 1]
    data3 = [10, 9, 8, 7]

    ax.plot(data1, data2, "x", c="green", label="Data 1")
    ax.plot(data1, data3, "o", c="red", label="Data 2")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    graph_linear_regression(118.5264873504491, 0.0954029565126554)
