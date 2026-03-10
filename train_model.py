"""
train_model.py — Disease Image Classification (MobileNetV2 Transfer Learning)

Folder structure:
ML/
 ├── train/
 │    ├── Healthy/
 │    ├── Powdery/
 │    └── Rust/
 ├── Test/
 │    ├── Healthy/
 │    ├── Powdery/
 │    └── Rust/
 └── train_model.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS_WARMUP = 8
EPOCHS_FINETUNE = 12
UNFREEZE_LAST_N = 80
USE_CLASS_WEIGHTS = True
RANDOM_SEED = 42
MODEL_PATH = "disease_model.h5"
LABELS_JSON = "labels.json"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def find_folder_case_insensitive(base, name):
    """Finds a folder ignoring case (train, Train, TRAIN...)."""
    for variant in [name, name.lower(), name.upper(), name.capitalize()]:
        path = os.path.join(base, variant)
        if os.path.isdir(path):
            return path
    return None


base_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = find_folder_case_insensitive(base_dir, "train")
test_dir = find_folder_case_insensitive(base_dir, "Test")

if not train_dir or not test_dir:
    raise FileNotFoundError("❌ train/ and Test/ folders not found inside ML directory!")

print(f"✅ Using train folder: {train_dir}")
print(f"✅ Using test folder:  {test_dir}")

train_datagen = ImageDataGenerator(
    validation_split=VAL_SPLIT,
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=(0.8, 1.2),
    shear_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=RANDOM_SEED
)
val_gen = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    seed=RANDOM_SEED
)
test_gen = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_gen.num_classes
class_names = list(train_gen.class_indices.keys())
print("\n🔖 Class mapping:", train_gen.class_indices)


with open(LABELS_JSON, "w") as f:
    json.dump({"class_indices": train_gen.class_indices}, f, indent=2)
print(f"💾 Saved labels to {LABELS_JSON}")


class_weight = None
if USE_CLASS_WEIGHTS:
    y_train_labels = train_gen.classes
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print(f"⚖️ Class Weights: {class_weight}")


base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet")
base_model.trainable = False  

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=1),
]

print("\n🔥 Phase 1: Training top layers (base frozen)")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_WARMUP,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


for layer in base_model.layers[-UNFREEZE_LAST_N:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

print("\n🎯 Phase 2: Fine-tuning last layers...")
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


history = {
    "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
    "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
    "loss": history1.history["loss"] + history2.history["loss"],
    "val_loss": history1.history["val_loss"] + history2.history["val_loss"],
}


model.save(MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")


print("\n🧪 Evaluating on Test set...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"✅ Test Accuracy: {test_acc*100:.2f}% | Loss: {test_loss:.4f}")

y_prob = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_gen.classes

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history["accuracy"], label="Train Acc")
plt.plot(history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, class_names)


mis_idx = np.where(y_true != y_pred)[0]
print(f"\n🔎 Showing first {min(12, len(mis_idx))} misclassified samples...")
for i in mis_idx[:12]:
    img_path = test_gen.filepaths[i]
    img = plt.imread(img_path)
    plt.figure()
    plt.imshow(img)
    plt.title(f"True: {class_names[y_true[i]]} | Pred: {class_names[y_pred[i]]}")
    plt.axis('off')
    plt.show()
model.export("saved_model") 
