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


d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
"""
Accessing
"""

# access an element
d[2]
# 20

# access final element
d[-1]
# 10

# access all
d
# [0. 1. 2. 3. 4. 5. 6. 7. 8.]

# access a range
d[2:7:1]  # indexes 2-7 stepped by 1
# [2. 3. 4. 5. 6.]

# access a range
d[2:7:2]  # indexes 2-7 stepped by 2
# [2. 4. 6]

# access index 3 and above
d[3:]
# [3. 4. 5. 6. 7. 8.]

# access all below 3
d[:3]
# [0. 1. 2.]

"""
Math
"""

# negate all elements
-d
# [ 0. -1. -2. ...]

# add all elements
np.sum(d)
# 36

# get the mean
np.mean(d)

# affect all by 2
d**2
# [0. 1. 4. 9. 16. ...]

"""
Binary operators will work element to element! 
Vectors must be of the same size!
"""

y = np.array([0, 1, 2, 3])
z = np.array([0, 0, 1, 5])

q = y + z
# q = [0. 1. 3. 8.]
