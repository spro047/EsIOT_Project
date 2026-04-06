import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ==========================================
# Step 1: GPU Configuration
# ==========================================
def setup_gpu():
    """Configures TensorFlow to use maximum GPU efficiency."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU successfully configured. Found {len(physical_devices)} GPUs.")
        except RuntimeError as e:
            print("⚠️ GPU Setup Error:", e)
    else:
        print("⚠️ No GPU found. Training will use CPU, which may be significantly slower.")

setup_gpu()

# ==========================================
# Step 2: Data Preprocessing (80/20 Split)
# ==========================================
# Ensure this path matches the 'Original' or 'Augmentated' dataset location relative to where this runs
DATASET_DIR = '../Original/'

def load_data(dataset_path=DATASET_DIR, target_size=(256, 256), batch_size=32):
    """Loads and splits the dataset into 80% Training and 20% Validation."""
    print("Preparing image generators...")
    
    datagen = ImageDataGenerator(
        rescale=1./255,      # Normalize all pixel intensities to [0, 1]
        validation_split=0.2 # 20% validation split
    )

    print("Loading Training data:")
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    print("Loading Validation data:")
    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

# ==========================================
# Step 3 & 4: Build & Compile the CNN Model
# ==========================================
def build_cnn(input_shape=(256, 256, 3), num_classes=39):
    """Defines a deep CNN architecture capable of classifying plant species and disease."""
    print("Building the CNN architecture...")
    model = models.Sequential([
        # Convolutional Block 1 - Extracts foundational edges and textures
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 2 - Extracts specific shapes and spotting/mold visuals
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 3 - Extracts complex high-level disease characteristics
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten feature maps into a 1D tensor
        layers.Flatten(),
        
        # Fully Connected (Dense) Layers for mathematical reasoning
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Standard dropout prevents overfitting to the training data
        
        # Output layer with exactly 39 endpoints for simultaneous species/disease classification
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the final layout using an optimizer, loss metric, and tracking
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    
    # 1. Evaluate your dataset directory check
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Dataset directory '{DATASET_DIR}' not found.")
        print("Please ensure the 'Original' folder is present in the parent directory.")
        # We allow execution to proceed to see the model summary even if data is missing right now.
    
    # 2. Build Model
    model = build_cnn()
    model.summary()
    
    # 3. Warning on running fit() blindly
    print("\n" + "="*50)
    print("Ready to train! To initiate training, you would execute the following:")
    print("train_gen, val_gen = load_data()")
    print("history = model.fit(train_gen, validation_data=val_gen, epochs=10)")
    print("="*50 + "\n")
