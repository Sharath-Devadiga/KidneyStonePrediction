"""
Kidney Stone Detection - Image Classification Model Training
This script trains a CNN model to detect kidney stones from CT/X-ray images
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning & Image Processing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class KidneyStoneImageModel:
    """
    Complete pipeline for training kidney stone detection model
    """
    
    def __init__(self, img_size=(150, 150), batch_size=32):
        """
        Initialize the model trainer
        
        Args:
            img_size: Target size for images (width, height)
            batch_size: Batch size for training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Paths
        self.train_dir = "./CT_images/Train"
        self.test_dir = "./CT_images/Test"
        self.model_save_path = "./models/kidney_stone_cnn_model.h5"
        self.model_save_path_keras = "./models/kidney_stone_cnn_model.keras"
        
        print("=" * 70)
        print("Kidney Stone Image Classification Model Trainer")
        print("=" * 70)
        print(f"Image Size: {img_size}")
        print(f"Batch Size: {batch_size}")
        print(f"Train Directory: {self.train_dir}")
        print(f"Test Directory: {self.test_dir}")
        print("=" * 70)
    
    def prepare_data_generators(self):
        """
        Create data generators with augmentation for training
        """
        print("\n📊 Preparing Data Generators...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # 20% for validation
        )
        
        # Test data generator (only rescaling, no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        # Create validation generator
        self.validation_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=True
        )
        
        # Create test generator if test directory exists
        if os.path.exists(self.test_dir):
            self.test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )
        else:
            self.test_generator = None
            print("⚠️ Warning: Test directory not found")
        
        print(f"✅ Training samples: {self.train_generator.samples}")
        print(f"✅ Validation samples: {self.validation_generator.samples}")
        if self.test_generator:
            print(f"✅ Test samples: {self.test_generator.samples}")
        print(f"✅ Classes: {self.train_generator.class_indices}")
        
    def build_model(self):
        """
        Build the CNN architecture
        """
        print("\n🏗️ Building CNN Model...")
        
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall')]
        )
        
        print("\n📋 Model Architecture:")
        self.model.summary()
        
    def train_model(self, epochs=50):
        """
        Train the model with callbacks
        
        Args:
            epochs: Number of training epochs
        """
        print(f"\n🚀 Starting Training for {epochs} epochs...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            self.model_save_path_keras,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
    def evaluate_model(self):
        """
        Evaluate the model on validation/test data
        """
        print("\n📊 Evaluating Model Performance...")
        
        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall = self.model.evaluate(
            self.validation_generator,
            verbose=1
        )
        
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
        
        print("\n" + "=" * 70)
        print("VALIDATION METRICS")
        print("=" * 70)
        print(f"📈 Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"📉 Loss:      {val_loss:.4f}")
        print(f"🎯 Precision: {val_precision:.4f}")
        print(f"🔍 Recall:    {val_recall:.4f}")
        print(f"⭐ F1-Score:  {val_f1:.4f}")
        print("=" * 70)
        
        # Evaluate on test set if available
        if self.test_generator:
            test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
                self.test_generator,
                verbose=1
            )
            
            test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
            
            print("\n" + "=" * 70)
            print("TEST METRICS")
            print("=" * 70)
            print(f"📈 Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"📉 Loss:      {test_loss:.4f}")
            print(f"🎯 Precision: {test_precision:.4f}")
            print(f"🔍 Recall:    {test_recall:.4f}")
            print(f"⭐ F1-Score:  {test_f1:.4f}")
            print("=" * 70)
    
    def plot_training_history(self):
        """
        Plot training and validation metrics
        """
        print("\n📊 Generating Training History Plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = './models/training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {plot_path}")
        plt.show()
    
    def save_model(self):
        """
        Save the trained model
        """
        print("\n💾 Saving Model...")
        
        # Save in both formats
        self.model.save(self.model_save_path_keras)  # Keras format (recommended)
        self.model.save(self.model_save_path)  # H5 format (legacy)
        
        print(f"✅ Model saved to:")
        print(f"   - {self.model_save_path_keras} (Keras format)")
        print(f"   - {self.model_save_path} (H5 format)")
        
        # Save model info
        model_info = {
            'img_size': self.img_size,
            'classes': self.train_generator.class_indices,
            'training_date': datetime.now().isoformat(),
            'total_params': self.model.count_params()
        }
        
        import json
        info_path = './models/image_model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"✅ Model info saved to: {info_path}")
    
    def run_full_pipeline(self, epochs=50):
        """
        Run the complete training pipeline
        
        Args:
            epochs: Number of training epochs
        """
        print("\n" + "=" * 70)
        print("STARTING FULL TRAINING PIPELINE")
        print("=" * 70)
        
        # Step 1: Prepare data
        self.prepare_data_generators()
        
        # Step 2: Build model
        self.build_model()
        
        # Step 3: Train model
        self.train_model(epochs=epochs)
        
        # Step 4: Evaluate model
        self.evaluate_model()
        
        # Step 5: Plot training history
        self.plot_training_history()
        
        # Step 6: Save model
        self.save_model()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📁 Model files saved in: ./models/")
        print(f"📊 Use the saved model for predictions in your API")
        print("=" * 70)


def main():
    """
    Main execution function
    """
    print("\n🏥 Kidney Stone Detection - Image Model Training 🏥\n")
    
    # Create trainer instance
    trainer = KidneyStoneImageModel(
        img_size=(150, 150),  # Image size for training
        batch_size=32          # Batch size
    )
    
    # Run full training pipeline
    # Adjust epochs as needed (50 is a good starting point)
    trainer.run_full_pipeline(epochs=50)
    
    print("\n🎉 All done! Your model is ready for deployment.")
    print("💡 Next steps:")
    print("   1. Test the model with sample images")
    print("   2. Integrate with FastAPI (main.py)")
    print("   3. Connect to your web application")


if __name__ == "__main__":
    main()
