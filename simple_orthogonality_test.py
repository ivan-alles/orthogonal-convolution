import numpy as np
import tensorflow as tf

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

H = 2
W = 3
INPUTS = 4
OUTPUTS = 5

kernel = np.ones((H, W, INPUTS, OUTPUTS), dtype=np.float32)

# Make 0th and 1st kernels orthogonal

kernel[:, :, :, 0] = np.array([
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 7, 8, 6],
    ],
    [
        [5, 4, 3, 2],
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
])

s = np.sum(kernel[:, :, :, 0]) - 1

kernel[0, 0, 0, 1] = -s

conv = tf.keras.layers.Conv2D(
    OUTPUTS,
    (H, W),
    kernel_initializer=tf.keras.initializers.Constant(kernel),
)

model = tf.keras.Sequential([tf.keras.layers.Input(shape=(100, 100, INPUTS)), conv])
print(conv.kernel.shape)
print(conv.kernel.numpy())

d = tf.tensordot(conv.kernel, conv.kernel, [[0, 1, 2], [0, 1, 2]])
print(d)