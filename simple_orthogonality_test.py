import numpy as np

# For a conv2d  I -> O tensorflow keeps the kernel in tensors of shape H x W x I x O.

# Assume the kernels are in the columns (in general in the first dimensions).

# Non-orthogonal columns.
a = np.array([
    [1,     2,   3],
    [10,   20,  30],
    [100, 200, 300]
])

p1 = np.dot(a.T, a)
print(p1)

p2 = np.tensordot(a, a, axes=[[0], [0]])
print(p2)

# First 2 columns are orthogonal.
a = np.array([
    [1,  -2,   1],
    [0,   0,   1],
    [2,   1,   1]
])

p1 = np.dot(a.T, a)
print(p1)

p2 = np.tensordot(a, a, axes=[[0], [0]])
print(p2)

# All columns are orthogonal.
a = np.array([
    [1,  -2,   0],
    [0,   0,   10],
    [2,   1,   0]
])

p1 = np.dot(a.T, a)
print(p1)

p2 = np.tensordot(a, a, axes=[[0], [0]])
print(p2)