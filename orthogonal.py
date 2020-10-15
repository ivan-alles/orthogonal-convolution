"""
A baseline model to compare with experimental ones.
See https://www.tensorflow.org/tutorials/images/cnn
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import utils

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


utils.show_images(train_images, train_labels)

conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
conv2 = layers.Conv2D(64, (3, 3), activation='relu')
conv3 = layers.Conv2D(64, (3, 3), activation='relu')

model = models.Sequential()
model.add(conv1)
model.add(layers.MaxPooling2D((2, 2)))
model.add(conv2)
model.add(layers.MaxPooling2D((2, 2)))
model.add(conv3)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

def orthogonality(x):
    d = tf.tensordot(x, x, [[0, 1, 2], [0, 1, 2]])
    size = d.shape[0]
    num_elements = size * (size - 1) / 2
    o = tf.norm(tf.math.multiply(d, (1 - tf.eye(size)))) / num_elements
    return o

class OrthogonalLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    classification_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    o1 = orthogonality(conv1.kernel)
    o2 = orthogonality(conv2.kernel)
    o3 = orthogonality(conv3.kernel)
    return classification_loss + (o1 + o2 + o3) * 1000

model.compile(optimizer='adam',
              loss=OrthogonalLoss(),
              metrics=['accuracy'])

print(orthogonality(conv1.kernel),
      orthogonality(conv2.kernel),
      orthogonality(conv3.kernel))

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

utils.show_training_history(history)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(f'Test accuracy {test_acc}')

print(orthogonality(conv1.kernel),
      orthogonality(conv2.kernel),
      orthogonality(conv3.kernel))

plt.show()