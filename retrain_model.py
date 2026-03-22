# -*- coding: utf-8 -*-
"""
CelebMatch - Model Retraining & Improvement Script
Continues training the best saved model with enhanced strategies to achieve 80-90% accuracy

This script:
  1. Loads the best previously trained model (celebrity_model_best.h5)
  2. Applies aggressive data augmentation strategies
  3. Fine-tunes with optimized hyperparameters
  4. Monitors accuracy in real-time and stops when reaching 80-90%
  5. Saves improved model as the new best when target is reached

Usage:
    python retrain_model.py
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
)

# Force UTF-8 on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH = "Celebrity Faces Dataset"
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 16  # Reduced for better gradient estimation
EPOCHS       = 50  # Extended training
LR_START     = 1e-4
LR_MIN       = 1e-7
TARGET_ACC   = 0.80  # Stop when reaching 80% accuracy and above
BEST_MODEL   = "celebrity_model_best.h5"
FINAL_MODEL  = "celebrity_model.h5"
INFO_FILE    = "model_info.json"

# ── Custom Callback for Accuracy Monitoring ──────────────────────────────────
class AccuracyMonitor(Callback):
    """Monitors validation accuracy and stops training when target is reached"""
    def __init__(self, target_acc=0.80, patience=3):
        super().__init__()
        self.target_acc = target_acc
        self.patience = patience
        self.best_acc = 0
        self.patience_count = 0
        self.target_reached_epoch = None
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy', 0)
        
        if val_acc >= self.target_acc:
            if self.target_reached_epoch is None:
                self.target_reached_epoch = epoch + 1
                print(f"\n✓ TARGET ACCURACY REACHED: {val_acc*100:.2f}% (Epoch {epoch+1})")
                self.patience_count = 0
            else:
                self.patience_count += 1
                if self.patience_count >= self.patience:
                    print(f"\n✓ Maintaining target accuracy for {self.patience} epochs. Stopping training.")
                    self.model.stop_training = True
        else:
            self.patience_count = 0

# ── Data Generators with Enhanced Augmentation ──────────────────────────────
print("=" * 70)
print("  CelebMatch - Model Retraining with Enhanced Accuracy (80-90%)")
print("=" * 70)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    # Aggressive augmentation for better generalization
    horizontal_flip=True,
    rotation_range=20,  # Increased from 15
    brightness_range=[0.70, 1.30],  # Increased from [0.75, 1.25]
    zoom_range=0.20,  # Increased from 0.15
    width_shift_range=0.15,  # Increased from 0.1
    height_shift_range=0.15,  # Increased from 0.1
    shear_range=0.10,  # Increased from 0.05
    fill_mode='nearest',
    vertical_flip=False,  # Don't flip faces vertically
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

print("\n[INFO] Loading and preparing data ...")
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

print(f"\n[OK] Classes ({NUM_CLASSES}): {', '.join(CLASS_NAMES)}")
print(f"     Training samples   : {train_data.samples}")
print(f"     Validation samples : {val_data.samples}\n")

# ── Load Best Previously Trained Model ───────────────────────────────────────
print(f"[INFO] Loading best previously trained model: {BEST_MODEL}")
if not os.path.exists(BEST_MODEL):
    print(f"[ERROR] {BEST_MODEL} not found! Please train the model first using train_model.py")
    sys.exit(1)

model = tf.keras.models.load_model(BEST_MODEL)

# Load previous model info to get baseline
if os.path.exists(INFO_FILE):
    with open(INFO_FILE, 'r', encoding='utf-8') as f:
        prev_info = json.load(f)
        prev_acc = prev_info.get('evaluation', {}).get('val_accuracy', 0)
        prev_epochs = prev_info.get('model_params', {}).get('epochs_trained', 0)
        print(f"[OK] Previous best accuracy: {prev_acc*100:.2f}%")
        print(f"[OK] Previous epochs trained: {prev_epochs}")

print(f"[OK] Model loaded successfully!")
print(f"[INFO] Model has {len(model.layers)} layers\n")

# ── Callbacks ────────────────────────────────────────────────────────────────
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
            patience=8,  # Increased patience
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,  # Increased patience before reducing LR
            min_lr=LR_MIN,
            verbose=1,
        ),
        AccuracyMonitor(
            target_acc=TARGET_ACC,
            patience=3,  # Stop if maintaining target for 3 epochs
        ),
    ]

# ── Retraining: Continue improving the model ────────────────────────────────
print(f">> Retraining Phase - Improving model accuracy (target: {TARGET_ACC*100:.0f}%-90%)")
print(f"   Starting with enhanced data augmentation and optimized parameters ...\n")

model.compile(
    optimizer=Adam(learning_rate=LR_START),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_retrain = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=make_callbacks(),
    verbose=1,
)

# ── Evaluate Final Model ──────────────────────────────────────────────────────
print("\n>> Evaluating retrained model on validation set ...")
val_loss, val_acc = model.evaluate(val_data, verbose=1)
print(f"\n   Final Val Loss     : {val_loss:.4f}")
print(f"   Final Val Accuracy : {val_acc * 100:.2f}%")

# ── Determine if target is reached ───────────────────────────────────────────
if val_acc >= TARGET_ACC:
    print(f"\n✓ SUCCESS! Target accuracy ({TARGET_ACC*100:.0f}%) reached!")
    print(f"   Achieved accuracy: {val_acc*100:.2f}%")
else:
    print(f"\n⚠ Target accuracy ({TARGET_ACC*100:.0f}%) not yet reached.")
    print(f"   Current accuracy: {val_acc*100:.2f}%")
    print(f"   Consider running more epochs or adjusting hyperparameters.")

# ── Save Final Model ──────────────────────────────────────────────────────────
model.save(FINAL_MODEL)
print(f"\n[OK] Final model saved -> {FINAL_MODEL}")
print(f"[OK] Best model saved  -> {BEST_MODEL}")

# ── Update Model Info JSON ────────────────────────────────────────────────────
total_epochs = len(history_retrain.history.get("accuracy", []))

model_info = {
    "model_params": {
        "epochs_trained":    total_epochs,
        "epochs_head":       15,  # From original training
        "epochs_finetune":   15,  # From original training
        "epochs_retrain":    total_epochs,
        "batch_size":        BATCH_SIZE,
        "img_size":          list(IMG_SIZE),
        "optimizer":         "Adam",
        "lr_start":          LR_START,
        "lr_min":            LR_MIN,
        "num_classes":       NUM_CLASSES,
        "backbone":          "MobileNetV2 (ImageNet pretrained)",
        "augmentation": [
            "horizontal_flip: True",
            "rotation_range: 20°",
            "brightness_range: [0.70, 1.30]",
            "zoom_range: 0.20",
            "width_shift_range: 0.15",
            "height_shift_range: 0.15",
            "shear_range: 0.10",
        ],
        "training_strategy": "Continued fine-tuning from best saved model",
    },
    "class_names": CLASS_NAMES,
    "evaluation": {
        "val_loss":     round(float(val_loss), 4),
        "val_accuracy": round(float(val_acc), 4),
        "target_reached": val_acc >= TARGET_ACC,
    },
    "training_history": {
        "accuracy":     [round(float(v), 4) for v in history_retrain.history.get("accuracy", [])],
        "val_accuracy": [round(float(v), 4) for v in history_retrain.history.get("val_accuracy", [])],
        "loss":         [round(float(v), 4) for v in history_retrain.history.get("loss", [])],
        "val_loss":     [round(float(v), 4) for v in history_retrain.history.get("val_loss", [])],
    },
}

with open(INFO_FILE, "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=2)

print(f"[OK] Model info updated -> {INFO_FILE}")
print("\n" + "=" * 70)
print("  Retraining Complete!")
if val_acc >= TARGET_ACC:
    print(f"  Model accuracy: {val_acc*100:.2f}% ✓ (Target: {TARGET_ACC*100:.0f}%+)")
else:
    print(f"  Model accuracy: {val_acc*100:.2f}% (Target: {TARGET_ACC*100:.0f}%+) - Consider retraining again")
print("=" * 70)
