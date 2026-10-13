import json
import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------------
# 1. Project configuration
# -----------------------------------

DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "waste_classifier.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------------
# 2. Check the dataset
# -----------------------------------

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        "Dataset folder not found. "
        "Make sure dataset is inside WasteClassification."
    )

class_folders = [
    name
    for name in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, name))
]

if len(class_folders) < 2:
    raise ValueError(
        "At least two category folders with images are required."
    )

print("Waste categories found:", class_folders)


# -----------------------------------
# 3. Load training and validation data
# -----------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_dataset.class_names
number_of_classes = len(class_names)

print("\nCategories used by the model:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

# Save category names for the prediction application.
with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)


# -----------------------------------
# 4. Improve data-loading performance
# -----------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# -----------------------------------
# 5. Data augmentation
# -----------------------------------

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# -----------------------------------
# 6. Build the CNN model
# -----------------------------------

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    # First convolution block
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(
        number_of_classes,
        activation="softmax"
    )
])


# -----------------------------------
# 7. Compile the model
# -----------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------------
# 8. Train the model
# -----------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nStarting CNN model training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15,
    callbacks=[early_stopping]
)


# -----------------------------------
# 9. Evaluate the model
# -----------------------------------

validation_loss, validation_accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print(
    f"\nValidation accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Validation loss: {validation_loss:.4f}"
)


# -----------------------------------
# 10. Save the trained model
# -----------------------------------

model.save(MODEL_PATH)

print("\nTraining completed successfully!")
print("Model saved at:", MODEL_PATH)
print("Class names saved at:", CLASS_NAMES_PATH)
print("\nYou can now run app.py.")