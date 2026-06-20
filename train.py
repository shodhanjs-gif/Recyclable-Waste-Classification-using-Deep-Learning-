import tensorflow as tf
"""
Manual Training Script for Waste Detection Model

Usage:
    python train.py

Prerequisites:
    Ensure 'dataset' directory exists with subfolders:
    - dataset/plastic
    - dataset/paper
    - dataset/metal
    - dataset/glass
    - dataset/organic

    Add your images to these folders before running.
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Configuration
DATASET_DIR = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

def train_model():
    # Check if dataset directory exists and has images
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        subset='validation'
    )

    if train_generator.samples == 0:
        print("No images found in dataset. Please add images to 'backend/dataset/' subfolders.")
        return

    # Base Model (Transfer Learning)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(CLASSES), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile with initial learning rate
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    checkpoint = ModelCheckpoint(
        'waste_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Initial Training
    print("Starting initial training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        epochs=15, # Increased initial epochs
        callbacks=[checkpoint, early_stopping]
    )

    # Fine-Tuning
    print("Starting fine-tuning...")
    # Unfreeze the top layers of the model
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100 # Freeze the first 100 layers
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile with a lower learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train again
    total_epochs = 25 # 15 + 10
    
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        callbacks=[checkpoint, early_stopping]
    )

    print("Training complete.")

if __name__ == "__main__":
    train_model()
