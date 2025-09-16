import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import copy

import const, graphing
from get_scaled_features import get_training_data


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = np.float32(0)

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2

    total_cost = cost / (2 * m)

    return total_cost


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = np.float32(0)

    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        err = f_wb - y[i]

        for j in range(n):
            dj_dw[j] += err * X[i, j]

        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw


def gradient_descent(
    X, y, w_init, b_init, num_iters, alpha, cost_function, gradient_function
):
    J_hist = []
    p_hist = []

    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_hist.append(cost_function(X, y, w, b))
        p_hist.append([w, b])

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration: {i:4},\t Cost: {J_hist[-1]},\t w: {w},\t b: {b:.5f}")

    return w, b, J_hist, p_hist


# def get_training_data():

#     X = []
#     y = []

#     with open(const.TRAINING_SET, "r") as f:
#         reader = csv.DictReader(f)

#         for row in reader:
#             x_set = [
#                 int(row["1stFlrSF"]),
#                 int(row["YearBuilt"]),
#                 int(row["OverallCond"]),
#             ]
#             X.append(x_set)
#             y.append(int(row["SalePrice"]))

#     return np.array(X), np.array(y)


if __name__ == "__main__":
    X_train, y_train = get_training_data()
    m, n = X_train.shape
    w_init = np.array([34.43899115, 79.60193739, 1.20144644])
    b_init = np.float64(1.65769)
    iterations = 100000
    alpha_init = np.float64(100.0e-10)

    w_final, b_final, J_hist, p_hist = gradient_descent(
        X_train,
        y_train,
        w_init,
        b_init,
        iterations,
        alpha_init,
        compute_cost,
        compute_gradient,
    )

    print(f"w: {w_final}\nb: {b_final}\n")

    graphing.graph_linear_regression(w_final, b_final)
    graphing.graph_J(J_hist)

    plt.show()
