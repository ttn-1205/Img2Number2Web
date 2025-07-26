import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load the pretrained base model
initial_model = tf.keras.models.load_model('Initial_MNIST.keras')

# Directories for fine-tuning dataset
train_dir = "D:/OneDrive/Desktop/Digits_for_Finetuning/train"
val_dir = "D:/OneDrive/Desktop/Digits_for_Finetuning/val"
test_dir = "D:/OneDrive/Desktop/Digits_for_Finetuning/test"

# Constants
IMG_HEIGHT, IMG_WIDTH = 28, 28
NUM_CLASSES = 10
INITIAL_LR = 0.00003

# Data generators
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
    batch_size=32, class_mode='sparse', shuffle=True
)
val_data = val_gen.flow_from_directory(
    val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
    batch_size=64, class_mode='sparse', shuffle=True
)
test_data = test_gen.flow_from_directory(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
    batch_size=128, class_mode='sparse', shuffle=False
)

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)

# Construct new model using base layers + new trainable layers
new_model = tf.keras.models.Sequential([
    initial_model.layers[0],  # Input layer
    initial_model.layers[1],  # First Conv layer (frozen)

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', name = "conv1"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "pool1"),
    tf.keras.layers.Dropout(0.25, name = "dropout1"),

    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', name = "conv2"),
    tf.keras.layers.MaxPooling2D((2, 2), name = "pool2"),
    tf.keras.layers.Dropout(0.25, name = "dropout2"),

    tf.keras.layers.Flatten(name = "flatten"),
    tf.keras.layers.Dense(128, activation = 'relu', name = "dense"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation = 'softmax', name = "output")
])

new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning the model
print("\n--- Starting Fine-tuning ---")
history = new_model.fit(
    train_data,
    epochs=500,
    validation_data=val_data,
    callbacks=[early_stopping]
)
print("--- Fine-tuning Complete ---")

# Save the model
new_model.save('Finetuned_MNIST.keras')

# Evaluate on test set
if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
    print("\n--- Evaluating Fine-tuned Model on Test Data ---")
    loss, acc = new_model.evaluate(test_data)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# Display per-digit accuracy
def display_per_digit_accuracy(model, generator, name="Dataset"):
    print(f"\n--- Per-Digit Accuracy for {name} ---")
    generator.reset()
    steps = int(np.ceil(generator.n / generator.batch_size))
    preds = model.predict(generator, steps=steps, verbose=0)
    predicted_classes = np.argmax(preds, axis=1)
    true_classes = generator.classes

    if len(predicted_classes) != len(true_classes):
        print("Mismatch in prediction and label counts.")
        return

    correct = np.zeros(NUM_CLASSES, dtype=int)
    total = np.zeros(NUM_CLASSES, dtype=int)
    label_map = {v: k for k, v in generator.class_indices.items()}

    for true, pred in zip(true_classes, predicted_classes):
        total[true] += 1
        if true == pred:
            correct[true] += 1

    for i in range(NUM_CLASSES):
        label = label_map.get(i, str(i))
        if total[i] > 0:
            acc = (correct[i] / total[i]) * 100
            print(f"Digit {label}: {acc:.2f}% ({correct[i]}/{total[i]})")
        else:
            print(f"Digit {label}: No samples.")

    print(f"\n--- Classification Report ({name}) ---")
    print(classification_report(true_classes, predicted_classes, target_names=[str(i) for i in range(NUM_CLASSES)], zero_division=0))

    print(f"\n--- Confusion Matrix ({name}) ---")
    print(confusion_matrix(true_classes, predicted_classes))

# Re-create val generator (non-shuffled) for consistent evaluation
val_data_eval = val_gen.flow_from_directory(
    val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
    batch_size=64, class_mode='sparse', shuffle=False
)

# Run reports
display_per_digit_accuracy(new_model, val_data_eval, "Validation Set")
display_per_digit_accuracy(new_model, test_data, "Test Set")
