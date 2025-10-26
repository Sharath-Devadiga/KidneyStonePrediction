# ----------------------------
# Optimized CNN Training Script
# ----------------------------

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 100
TRAIN_DIR = "/content/drive/MyDrive/kidnet-ct/Train"
TEST_DIR = "/content/drive/MyDrive/kidnet-ct/Test"
MODEL_DIR = "/content/drive/MyDrive/kidnet-ct/Model"
MODEL_NAME = "kidney_stone_cnn_model_better.keras"
SEED = 42
INIT_LR = 8e-4

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# ----------------------------
# TF / GPU Settings
# ----------------------------
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

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

def prepare_datasets(train_dir, val_split=0.2):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode='binary',
        validation_split=val_split,
        subset='training',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode='binary',
        validation_split=val_split,
        subset='validation',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds = None
    if os.path.exists(TEST_DIR):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            TEST_DIR,
            label_mode='binary',
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

    # Augmentation as a layer
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.06, 0.06)
    ], name="data_augmentation")

    rescale = layers.Rescaling(1.0 / 255.0)

    def prepare(ds, training=False):
        ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            ds = ds.shuffle(1024)
        ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = prepare(train_ds, training=True)
    val_ds = prepare(val_ds)
    if test_ds:
        test_ds = prepare(test_ds)

    # steps per epoch (fallback if cardinality fails)
    try:
        train_steps = tf.data.experimental.cardinality(train_ds).numpy()
        val_steps = tf.data.experimental.cardinality(val_ds).numpy()
    except Exception:
        train_steps = max(1, sum([len(files) for r, d, files in os.walk(train_dir)]) // BATCH_SIZE)
        val_steps = max(1, int(0.2 * train_steps))

    return train_ds, val_ds, test_ds, train_steps, val_steps

# ----------------------------
# Model builder
# ----------------------------
def build_model(input_shape=(*IMG_SIZE, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

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
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return keras.Model(inputs, outputs, name="kidney_stone_cnn_better")

# ----------------------------
# Training pipeline
# ----------------------------
def train_pipeline(epochs=EPOCHS):
    train_ds, val_ds, test_ds, train_steps, val_steps = prepare_datasets(TRAIN_DIR)

    model = build_model()

    # Cosine LR schedule
    total_steps = epochs * train_steps
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INIT_LR,
        decay_steps=total_steps,
        alpha=0.01
    )

    base_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    model.summary()

    # Callbacks (Remove ReduceLROnPlateau to avoid LR conflict)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    print("Training finished.")

    # Evaluate
    print("Validation metrics:", model.evaluate(val_ds))
    if test_ds:
        print("Test metrics:", model.evaluate(test_ds))

    # Save final model & info
    try:
        model.save(MODEL_SAVE_PATH)
    except Exception as e:
        print("Could not save model:", e)

    info = {
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "total_params": model.count_params(),
        "training_date": datetime.now().isoformat()
    }
    with open(os.path.join(MODEL_DIR, "image_model_info_better.json"), "w") as f:
        json.dump(info, f, indent=2)

    return model, history

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    model, history = train_pipeline()
