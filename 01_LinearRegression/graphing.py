# Resources
#  + https://matplotlib.org/
#  + https://numpy.org/doc/

import matplotlib.pyplot as plt
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
    with open(const.TRAINING_SET, mode="r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            x_train.append(int(row["1stFlrSF"]))
            y_train.append(int(row["SalePrice"]))

    # Get Test Data
    with open(const.TEST_SET, "r") as file:
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
    print("Running from graphing.py")

    graph_linear_regression(152.498, 25)
    plt.show()
