"""
Optimized - Fast + Better Accuracy CNN training script
- Mixed precision (mixed_float16)
- XLA JIT enabled
- tf.data pipeline: cache + prefetch + augmentation
- Slightly deeper but efficient CNN
- Cosine learning-rate schedule (correct usage)
- Proper compilation after computing steps-per-epoch
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime

# ----------------------------
# Config (edit as needed)
# ----------------------------
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 100
TRAIN_DIR = "./CT_images/Train"  # update if necessary
TEST_DIR = "./CT_images/Test"    # optional
MODEL_DIR = "./models"
MODEL_NAME = "kidney_stone_cnn_model_better.keras"
SEED = 42
INIT_LR = 8e-4   # sensible initial LR for Adam with CosineDecay

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# ----------------------------
# TF / GPU Performance Settings
# ----------------------------
# Mixed precision (gives big speedups on modern GPUs like T4/P100/V100/A100)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Enable XLA JIT compiler (can speed up some models)
tf.config.optimizer.set_jit(True)

# Avoid grabbing all GPU memory on Colab
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("Warning: could not set memory growth:", e)

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# ----------------------------
# Data pipeline
# ----------------------------
AUTOTUNE = tf.data.AUTOTUNE

def prepare_datasets(train_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED, val_split=0.2):
    """
    Returns: train_ds, val_ds, test_ds (or None), train_steps (batches), val_steps (batches)
    """
    print("Preparing datasets with image_dataset_from_directory...")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode='binary',
        seed=seed,
        validation_split=val_split,
        subset='training',
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode='binary',
        seed=seed,
        validation_split=val_split,
        subset='validation',
        image_size=img_size,
        batch_size=batch_size
    )

    test_ds = None
    if os.path.exists(TEST_DIR):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            TEST_DIR,
            label_mode='binary',
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )

    # Augmentation (as a layer so it runs on GPU and inside the graph)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.06, 0.06)
    ], name="data_augmentation")

    rescale = layers.Rescaling(1.0 / 255.0)

    def prepare(ds, training=False):
        # rescale
        ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
        if training:
            # augmentation only during training
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            ds = ds.shuffle(1024)
        # cache in-memory (if dataset fits); remove or use filename if memory is limited
        ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = prepare(train_ds, training=True)
    val_ds = prepare(val_ds, training=False)
    if test_ds is not None:
        test_ds = prepare(test_ds, training=False)

    # compute steps per epoch (cardinality returns number of batches)
    try:
        train_steps = tf.data.experimental.cardinality(train_ds).numpy()
        val_steps = tf.data.experimental.cardinality(val_ds).numpy()
    except Exception:
        train_steps = None
        val_steps = None

    # fallback: compute from file counts if cardinality unknown
    if train_steps is None or train_steps <= 0:
        total_train_files = 0
        for class_dir in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            for f in os.listdir(class_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    total_train_files += 1
        train_steps = max(1, total_train_files // batch_size)
        val_steps = max(1, int(0.2 * train_steps))

    print(f"Prepared datasets: train_steps={train_steps}, val_steps={val_steps}")
    return train_ds, val_ds, test_ds, train_steps, val_steps

# ----------------------------
# Model builder (uncompiled)
# ----------------------------
def build_model(input_shape=(*IMG_SIZE, 3)):
    """
    Build (but do not compile) the model. Compilation is done in train_pipeline where
    we have steps-per-epoch to build LR schedule.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # slightly deeper block for better accuracy
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    # Output should be float32 for numeric stability with mixed precision
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = keras.Model(inputs, outputs, name="kidney_stone_cnn_better")
    return model

# ----------------------------
# Training pipeline
# ----------------------------
def train_pipeline(epochs=EPOCHS):
    # Prepare datasets
    train_ds, val_ds, test_ds, train_steps, val_steps = prepare_datasets(TRAIN_DIR)

    # Ensure we have sane steps
    if train_steps <= 0:
        raise ValueError("train_steps resolved to 0. Check your TRAIN_DIR and BATCH_SIZE.")

    # Build uncompiled model
    model = build_model()

    # Compute total steps for LR schedule
    total_steps = int(epochs * train_steps)

    # Cosine decay schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INIT_LR,
        decay_steps=total_steps,
        alpha=0.01
    )

    # Build optimizer and wrap with LossScaleOptimizer for mixed precision
    base_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)

    # Compile model now that we have optimizer/schedule
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )

    print("Training finished. Evaluating...")

    # Evaluate validation
    val_metrics = model.evaluate(val_ds, verbose=1)
    print("Validation metrics:", val_metrics)

    # Evaluate on test if provided
    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, verbose=1)
        print("Test metrics:", test_metrics)

    # Save best model (ModelCheckpoint already saved best weights)
    try:
        model.save(MODEL_SAVE_PATH)
        print("Model final saved to:", MODEL_SAVE_PATH)
    except Exception as e:
        print("Could not save final model:", e)

    # Save some info
    info = {
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "training_date": datetime.now().isoformat(),
        "total_params": model.count_params()
    }
    with open(os.path.join(MODEL_DIR, "image_model_info_better.json"), "w") as f:
        json.dump(info, f, indent=2)

    return model, history

# ----------------------------
# Run training
# ----------------------------
if __name__ == "__main__":
    model, history = train_pipeline(epochs=EPOCHS)
    print("Done.")
