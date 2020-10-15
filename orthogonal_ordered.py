import tensorflow as tf
import numpy as np

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
dense1 = layers.Dense(64, activation='relu')
dense2 = layers.Dense(10)

model = models.Sequential()
model.add(conv1)
model.add(layers.MaxPooling2D((2, 2)))
model.add(conv2)
model.add(layers.MaxPooling2D((2, 2)))
model.add(conv3)
model.add(layers.Flatten())
model.add(dense1)
model.add(dense2)

model.summary()

def orthogonality(layer):
    k = layer.kernel
    axes = list(range(len(k.shape) - 1))
    d = tf.tensordot(k, k, [axes, axes])
    size = d.shape[0]
    num_elements = size * size
    diag = np.power(10, np.array(list(range(size, 0, -1))) / size)
    target = tf.linalg.diag(diag.astype(np.float32))
    o = tf.norm(d - target) / num_elements
    return o

class OrthogonalLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    classification_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    c1 = orthogonality(conv1)
    c2 = orthogonality(conv2)
    c3 = orthogonality(conv3)
    return classification_loss + c1 * 1000 + c2 * 500 + c3 * 250

model.compile(optimizer='adam',
              loss=OrthogonalLoss(),
              metrics=['accuracy'])

print(orthogonality(conv1),
      orthogonality(conv2),
      orthogonality(conv3),
      orthogonality(dense1),
      orthogonality(dense2))

history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))

utils.show_training_history(history)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(f'Test accuracy {test_acc}')

print(orthogonality(conv1),
      orthogonality(conv2),
      orthogonality(conv3),
      orthogonality(dense1),
      orthogonality(dense2))

plt.show()