import tensorflow as tf
import numpy as np

# --- Load and preprocess MNIST dataset ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1] and reshape for CNN input
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]  # Shape: (60000, 28, 28, 1)
x_test = x_test[..., np.newaxis]    # Shape: (10000, 28, 28, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',  # Default learning rate: 0.001
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

# Save model
model.save('Initial_MNIST.keras') 