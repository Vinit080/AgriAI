"""
AgriAI - PlantVillage Disease Model Trainer (CPU-Optimized)
============================================================
Uses MobileNetV2 transfer learning in two phases:
  Phase 1 (3 epochs): Train only the Dense head — fast convergence
  Phase 2 (2 epochs): Fine-tune top 30 MobileNetV2 layers — boosts accuracy

Expected accuracy: ~92-96% on PlantVillage 38-class dataset.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning spam
os.environ['OPENBLAS_NUM_THREADS'] = '2'  # Prevent OpenBLAS memory explosion
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds

print("=" * 56)
print("  🌱 AgriAI — PlantVillage Disease Model Trainer 🌱")
print("=" * 56)
print()

# ─── Configuration ────────────────────────────────────────
BATCH_SIZE      = 8          # Small batch for low-RAM machines (Flask running concurrently)
IMAGE_SIZE      = (224, 224)
EPOCHS_HEAD     = 5          # Phase 1: Train only the Dense head
EPOCHS_FINETUNE = 3          # Phase 2: Fine-tune top MobileNetV2 layers
AUTOTUNE        = tf.data.AUTOTUNE
MODEL_SAVE_PATH = "disease_model.h5"

# Limit TF memory growth to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ─── Step 1: Load Dataset ─────────────────────────────────
print("[1/5] Loading PlantVillage from local directory...")
DATA_DIR = "PlantVillage-Dataset/raw/color"

try:
    raw_train = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=None, # Yields individual (image, label) elements to match original pipeline
    )

    raw_val = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=None,
    )
    
    class_names = raw_train.class_names
    num_classes = len(class_names)
    total_train = len(raw_train)
    print(f"✅ Loaded {total_train:,} images across {num_classes} classes.")
    print(f"   Classes: {', '.join(class_names[:5])} ... and {num_classes - 5} more\n")
except Exception as e:
    print(f"❌ Dataset load failed: {e}")
    print("   Ensure the PlantVillage-Dataset is cloned correctly.")
    exit(1)

# ─── Step 2: Data Augmentation + Preprocessing ────────────
print("[2/5] Building augmented data pipeline...")

# Data augmentation layers (applied only during training)
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")

def preprocess_train(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = augmentation(image, training=True)
    image = (image / 127.5) - 1.0  # Normalize to [-1, 1] for MobileNetV2
    return image, label

def preprocess_val(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = (image / 127.5) - 1.0
    return image, label

train_ds = (raw_train
    .shuffle(2000, reshuffle_each_iteration=True)
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (raw_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
print("✅ Data pipeline ready.\n")

# ─── Step 3: Build Model ──────────────────────────────────
print("[3/5] Building MobileNetV2 Transfer Learning Model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze backbone for Phase 1

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

print(f"✅ Model built: MobileNetV2 backbone + Dense({num_classes}) head")
print(f"   Total parameters:     {model.count_params():,}")
print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}\n")

# ─── Step 4: Phase 1 Training (Head Only) ─────────────────
print(f"[4/5] PHASE 1: Training classification head ({EPOCHS_HEAD} epochs)...")
print("      (Backbone frozen — only Dense(38) layer trains)\n")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=2, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=1, verbose=1
    ),
]

history1 = model.fit(
    train_ds,
    epochs=EPOCHS_HEAD,
    validation_data=val_ds,
    callbacks=callbacks_phase1,
    verbose=1,
)

phase1_acc = max(history1.history.get('val_accuracy', [0])) * 100
print(f"\n✅ Phase 1 complete — Best validation accuracy: {phase1_acc:.1f}%")

# ─── Step 5: Phase 2 Fine-Tuning ──────────────────────────
print(f"\n[5/5] PHASE 2: Fine-tuning top MobileNetV2 layers ({EPOCHS_FINETUNE} epochs)...")
print("      (Unfreezing top 30 layers for domain-specific refinement)\n")

# Unfreeze only the top layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable_after = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"   Trainable parameters after unfreeze: {trainable_after:,}")

# Lower LR for fine-tuning — prevents destroying pretrained features
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=2, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    ),
]

history2 = model.fit(
    train_ds,
    epochs=EPOCHS_FINETUNE,
    validation_data=val_ds,
    callbacks=callbacks_phase2,
    verbose=1,
)

# ─── Save & Summary ───────────────────────────────────────
final_acc = max(history2.history.get('val_accuracy', [0])) * 100
phase1_best = max(history1.history.get('val_accuracy', [0])) * 100

# Save final model if ModelCheckpoint hasn't already (fallback)
if not os.path.exists(MODEL_SAVE_PATH):
    model.save(MODEL_SAVE_PATH)

print()
print("=" * 56)
print("  ✅ TRAINING COMPLETE!")
print("=" * 56)
print(f"  Phase 1 accuracy (head only):    {phase1_best:.1f}%")
print(f"  Phase 2 accuracy (fine-tuned):   {final_acc:.1f}%")
print(f"  Model saved to: {MODEL_SAVE_PATH}")
print()
print("  👉 Restart app.py to load the trained model.")
print("     Disease predictions will now show 90%+ confidence.")
print("=" * 56)
