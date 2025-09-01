def calculate_cost(x, y, w, b):

    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost_sum += (f_wb[i] - y[i]) ** 2

    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost
