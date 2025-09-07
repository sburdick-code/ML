import numpy as np
import matplotlib.pyplot as plt
import math
import csv

import graphing, const


def compute_cost(x, y, w, b):
    m = len(x)
    cost = np.float32(0)

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2

    total_cost = cost / (2 * m)

    return total_cost


def compute_gradient(x, y, w, b):
    m = len(x)
    dj_dw = np.float32(0)
    dj_db = np.float32(0)

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    num_iters,
    alpha,
    cost_function,
    gradient_function,
):

    w = w_init
    b = b_init
    cost = np.float64(0)

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x_train, y_train, w, b)
        tmp_w = w - alpha * dj_dw
        tmp_b = b - alpha * dj_db

        w = tmp_w
        b = tmp_b

        if i < 100000:  # Keep memory safe :O
            J_hist.append(cost_function(x_train, y_train, w, b))
            p_hist.append([w, b])

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration: {i:4},\t Cost: {J_hist[-1]},\t w: {w:.5f}, b: {b:.5f}")

    return w, b, J_hist, p_hist


def get_training_data():
    x = []
    y = []

    with open(const.TRAINING_SET, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            x.append(int(row["1stFlrSF"]))
            y.append(int(row["SalePrice"]))

    return np.array(x), np.array(y)


if __name__ == "__main__":
    x_train, y_train = get_training_data()
    w_init = np.float64(0)
    b_init = np.float64(0)
    J_hist = []
    p_hist = []
    iterations = 10000
    alpha = np.float64(1.0e-10)

    w_final, b_final, J_hist, p_hist = gradient_descent(
        x_train,
        y_train,
        w_init,
        b_init,
        iterations,
        alpha,
        compute_cost,
        compute_gradient,
    )

    print(f"w:      \t {w_final}")
    print(f"b:      \t {b_final}")

    graphing.graph_linear_regression(w_final, b_final)
    graphing.graph_J(J_hist)
    graphing.graph_p_hist(p_hist, J_hist)

    plt.show()
