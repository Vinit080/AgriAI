"""
AgriAI - Seed Quality Model Trainer (v4 - EfficientNetB0, 3-class merged)
=========================================================================
Strategy:
  - Merge 4 raw classes into 3 quality tiers (more samples per class):
      Poor    = broken + silkcut   (~800 train images)
      Average = discolored         (~400 train images)
      Good    = pure               (~400 train images)
  - Resize to 128x128 for better spatial detail
  - Use EfficientNetB0 (designed for variable input sizes, works well at 128)
  - Two-phase training: head only → fine-tune top 30 layers

Expected accuracy: ~90%+
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import json
import shutil
import numpy as np
import tensorflow as tf

print("=" * 60)
print("  AgriAI - Seed Quality Model Trainer v4")
print("  (EfficientNetB0 | 128x128 | 3 merged quality classes)")
print("=" * 60)
print()

# ─── Configuration ────────────────────────────────────────
BATCH_SIZE        = 16
IMAGE_SIZE        = (128, 128)
EPOCHS_HEAD       = 8
EPOCHS_FINETUNE   = 5
MODEL_SAVE_PATH   = "model.h5"
CLASSES_SAVE_PATH = "seed_class_names.json"
TEMP_DATA_DIR     = "temp_seed_merged"

RAW_DATA_DIR = (
    r"D:\Fertilizer detection and seed quality 100%code"
    r"\Fertilizer detection and seed quality 100%code"
    r"\Fertilizer detection and seed quality 100%code"
    r"\Fertilizer detection\testing_set\Corn seed"
)

# Merge raw folder names → quality tier
CLASS_MERGE = {
    "broken":    "Poor",
    "silkcut":   "Poor",
    "discolored": "Average",
    "pure":      "High",
}

AUTOTUNE = tf.data.AUTOTUNE

# ─── Step 1: Build merged dataset directory ───────────────
print("[1/5] Merging class folders into quality tiers...")

if os.path.exists(TEMP_DATA_DIR):
    shutil.rmtree(TEMP_DATA_DIR)

merged_counts = {}
for src_folder, quality_label in CLASS_MERGE.items():
    src_path = os.path.join(RAW_DATA_DIR, src_folder)
    dst_path = os.path.join(TEMP_DATA_DIR, quality_label)
    if not os.path.exists(src_path):
        print(f"  WARNING: Skipping missing folder: {src_path}")
        continue
    os.makedirs(dst_path, exist_ok=True)
    files = os.listdir(src_path)
    for fname in files:
        src_file = os.path.join(src_path, fname)
        # Prefix filename with source folder to avoid name collisions
        dst_file = os.path.join(dst_path, f"{src_folder}_{fname}")
        if not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
    count = len(os.listdir(dst_path))
    merged_counts[quality_label] = count
    print(f"  {src_folder:12s} -> {quality_label:8s} | {len(files)} images")

print(f"\n  Merged class totals:")
for label, count in sorted(merged_counts.items()):
    print(f"    {label}: {count} images")
print()

# ─── Step 2: Load merged dataset ─────────────────────────
print("[2/5] Loading merged dataset...")

raw_train = tf.keras.utils.image_dataset_from_directory(
    TEMP_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=None,
)

raw_val = tf.keras.utils.image_dataset_from_directory(
    TEMP_DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=None,
)

class_names = raw_train.class_names  # Alphabetical: ['Average', 'High', 'Poor']
num_classes = len(class_names)
print(f"  Classes ({num_classes}): {class_names}")
print(f"  Training: {len(raw_train):,} | Validation: {len(raw_val):,}\n")

with open(CLASSES_SAVE_PATH, 'w') as f:
    json.dump(class_names, f)
print(f"  Class names saved to {CLASSES_SAVE_PATH}\n")

# ─── Step 3: Build data pipeline ─────────────────────────
print("[3/5] Building data pipeline...")

# EfficientNetB0 expects [0, 255] input (uses its own preprocessing internally)
def preprocess_train(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.05)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

def preprocess_val(image, label):
    return tf.cast(image, tf.float32), label

train_ds = (raw_train
    .shuffle(3000, reshuffle_each_iteration=True)
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (raw_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
print("  Pipeline ready.\n")

# ─── Step 4: Build EfficientNetB0 model ──────────────────
print("[4/5] Building EfficientNetB0 model...")

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

print(f"  Backbone: EfficientNetB0 (frozen)")
print(f"  Output: Dense({num_classes}) → {class_names}")
print(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}\n")

# ─── Phase 1: Train Head ──────────────────────────────────
print(f"[5/5] PHASE 1: Training head ({EPOCHS_HEAD} epochs)...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
]

history1 = model.fit(
    train_ds, epochs=EPOCHS_HEAD,
    validation_data=val_ds,
    callbacks=callbacks_p1, verbose=1,
)
phase1_acc = max(history1.history.get('val_accuracy', [0])) * 100
print(f"\n  Phase 1 best val accuracy: {phase1_acc:.1f}%")

# ─── Phase 2: Fine-Tune ───────────────────────────────────
print(f"\n  PHASE 2: Fine-tuning top 30 EfficientNetB0 layers ({EPOCHS_FINETUNE} epochs)...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1,
    ),
]

history2 = model.fit(
    train_ds, epochs=EPOCHS_FINETUNE,
    validation_data=val_ds,
    callbacks=callbacks_p2, verbose=1,
)
final_acc = max(history2.history.get('val_accuracy', [0])) * 100

if not os.path.exists(MODEL_SAVE_PATH):
    model.save(MODEL_SAVE_PATH)

# ─── Cleanup temp dir ─────────────────────────────────────
shutil.rmtree(TEMP_DATA_DIR, ignore_errors=True)

print()
print("=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
print(f"  Phase 1 accuracy (head):       {phase1_acc:.1f}%")
print(f"  Phase 2 accuracy (fine-tuned):  {final_acc:.1f}%")
print(f"  Model saved to: {MODEL_SAVE_PATH}")
print(f"  Classes saved to: {CLASSES_SAVE_PATH}")
print()
print("  Class index mapping:")
for i, name in enumerate(class_names):
    print(f"    {i}: {name}")
print()
print("  Restart app.py to load the new model.")
print("=" * 60)
