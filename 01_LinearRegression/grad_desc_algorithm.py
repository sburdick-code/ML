import matplotlib.pyplot as plt
import csv


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2

    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(
    x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function
):
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

    return w, b, J_history, p_history


def get_training_data(file_path):

    x = []
    y = []

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            x.append(int(row["1stFlrSF"]))
            y.append(int(row["SalePrice"]))

    return x, y


if __name__ == "__main__":
    data_set = "TrainingSets\LinearRegression\Set_01\HousingPrices_train.csv"
    x_train, y_train = get_training_data(data_set)
    w_init = 0
    b_init = 0
    iterations = 10000
    tmp_alpha = 1.0e-2

    w_final, b_final, J_hist, p_hist = gradient_descent(
        x_train,
        y_train,
        w_init,
        b_init,
        tmp_alpha,
        iterations,
        compute_cost,
        compute_gradient,
    )

    print(f"w:      \t {w_final}")
    print(f"b:      \t {b_final}")
    print(f"J_hist: \t {J_hist}")
    print(f"p_hist: \t {p_hist}")
