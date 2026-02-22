# -*- coding: utf-8 -*-
"""
CelebMatch - Celebrity Look-Alike CNN  (Transfer Learning with MobileNetV2)
Training Script

Run this script first to train the model and save:
  * celebrity_model.h5       - trained Keras model (best weights)
  * model_info.json          - class names, training history, evaluation metrics

Why Transfer Learning?
  The dataset has ~80 images per class (1040 total training images).
  Training a CNN from scratch on this few images yields ~random-chance accuracy.
  MobileNetV2 (pretrained on ImageNet 1.4M images) provides rich, pre-learned
  feature representations, giving very good accuracy even with small datasets.

Usage:
    python train_model.py
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# Force UTF-8 on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH = "Celebrity Faces Dataset"
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
EPOCHS_HEAD  = 15    # Phase 1: train only the top head
EPOCHS_FINE  = 15    # Phase 2: fine-tune last 30 layers of MobileNetV2
LR_HEAD      = 1e-3
LR_FINE      = 1e-4
BEST_MODEL   = "celebrity_model_best.h5"
FINAL_MODEL  = "celebrity_model.h5"
INFO_FILE    = "model_info.json"

# ── Data Generators ───────────────────────────────────────────────────────────
print("=" * 60)
print("  CelebMatch - Transfer Learning with MobileNetV2")
print("=" * 60)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    brightness_range=[0.75, 1.25],
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)

NUM_CLASSES = train_data.num_classes
CLASS_NAMES = [k for k, v in sorted(train_data.class_indices.items(), key=lambda x: x[1])]

print(f"\n[OK] Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"     Training samples   : {train_data.samples}")
print(f"     Validation samples : {val_data.samples}\n")

# ── Build Model (MobileNetV2 + Custom Head) ───────────────────────────────────
print("[INFO] Building MobileNetV2 transfer learning model ...")

base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Freeze base in Phase 1

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs, name="CelebMatch_MobileNetV2")
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks():
    return [
        ModelCheckpoint(
            BEST_MODEL,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

# ── Phase 1: Train the head only ─────────────────────────────────────────────
print(f"\n>> Phase 1 / 2 - Training classification head ({EPOCHS_HEAD} epochs) ...\n")
model.compile(
    optimizer=Adam(learning_rate=LR_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
history_head = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_HEAD,
    callbacks=make_callbacks(),
    verbose=1,
)

# ── Phase 2: Fine-tune – unfreeze last 30 layers of base ─────────────────────
print(f"\n>> Phase 2 / 2 - Fine-tuning MobileNetV2 last 30 layers ({EPOCHS_FINE} epochs) ...\n")
# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_HEAD + EPOCHS_FINE,
    initial_epoch=EPOCHS_HEAD,
    callbacks=make_callbacks(),
    verbose=1,
)

# ── Save Final Model ──────────────────────────────────────────────────────────
model.save(FINAL_MODEL)
print(f"\n[OK] Final model saved -> {FINAL_MODEL}")
print(f"[OK] Best model saved  -> {BEST_MODEL}")

# ── Merge histories ───────────────────────────────────────────────────────────
def merge_hist(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

combined = {
    "accuracy":     merge_hist(history_head, history_fine, "accuracy"),
    "val_accuracy": merge_hist(history_head, history_fine, "val_accuracy"),
    "loss":         merge_hist(history_head, history_fine, "loss"),
    "val_loss":     merge_hist(history_head, history_fine, "val_loss"),
}

# ── Evaluate on Validation Set ────────────────────────────────────────────────
print("\n>> Evaluating on validation set ...")
val_loss, val_acc = model.evaluate(val_data, verbose=1)
print(f"   Val Loss     : {val_loss:.4f}")
print(f"   Val Accuracy : {val_acc * 100:.2f}%\n")

# ── Save Model Info JSON ──────────────────────────────────────────────────────
model_info = {
    "model_params": {
        "epochs_trained":    len(combined["accuracy"]),
        "epochs_head":       EPOCHS_HEAD,
        "epochs_finetune":   EPOCHS_FINE,
        "batch_size":        BATCH_SIZE,
        "img_size":          list(IMG_SIZE),
        "optimizer":         "Adam",
        "lr_head":           LR_HEAD,
        "lr_finetune":       LR_FINE,
        "num_classes":       NUM_CLASSES,
        "backbone":          "MobileNetV2 (ImageNet pretrained)",
        "architecture": [
            "MobileNetV2 backbone (frozen in Phase 1,",
            "  last 30 layers fine-tuned in Phase 2)",
            "GlobalAveragePooling2D",
            "BatchNormalization",
            "Dense(256) -> ReLU -> Dropout(0.4)",
            "Dense(128) -> ReLU -> Dropout(0.3)",
            f"Dense({NUM_CLASSES}) -> Softmax",
        ],
    },
    "class_names": CLASS_NAMES,
    "evaluation": {
        "val_loss":     round(float(val_loss), 4),
        "val_accuracy": round(float(val_acc), 4),
    },
    "training_history": {
        "accuracy":     [round(float(v), 4) for v in combined["accuracy"]],
        "val_accuracy": [round(float(v), 4) for v in combined["val_accuracy"]],
        "loss":         [round(float(v), 4) for v in combined["loss"]],
        "val_loss":     [round(float(v), 4) for v in combined["val_loss"]],
    },
}

with open(INFO_FILE, "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=2)

print(f"[OK] Model info saved  -> {INFO_FILE}")
print("\n" + "=" * 60)
print("  Training complete! Run:  streamlit run app.py")
print("=" * 60)
