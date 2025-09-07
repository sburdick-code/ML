import numpy as np

# Create a np array of length n full of 0s
a = np.zeros(4)
print(f"np.zeros(4):  a={a}, \t shape={a.shape}, \t datatype={a.dtype},")

# Create a np array with values 0 to n (ex: [0. 1. 2. 3.])
b = np.arange(4.0)
print(f"np.arange(4): b={b}, \t shape={b.shape}, \t datatype={b.dtype},")

# Convert a python list into a np array
c = np.array([50, 40, 20, 10])
print(f"np.array():   c={c}, \t shape={c.shape}, \t datatype={c.dtype},")

# Dot product is much faster when vectorized!
x = np.dot(b, c)
print(f"np.dot(b, c): x={x}, \t\t\t shape={x.shape}, \t datatype={x.dtype},")
