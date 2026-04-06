# CNN Classification Model for Plant Leaf Diseases

## 1. Install Dependencies
Before running any code, you need to install TensorFlow. Run the following command in your Python environment or terminal:
```bash
pip install tensorflow
```

## 2. Using GPU for Maximum Performance
To ensure maximum use of your GPU during training, TensorFlow will automatically utilize available GPUs. You can verify GPU availability and configure dynamic memory allocation to prevent out-of-memory errors:

```python
import tensorflow as tf

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

# Maximize GPU usage efficiency
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

## 3. Dataset Context
Please refer to the [Dataset_Analysis.md](../Dataset_Analysis.md) file to understand the dataset structure. The dataset contains 39 distinct classes covering 14 crop species and their various diseases, healthy states, and a backgound class.

## 4. How to Make a CNN Model
A Convolutional Neural Network (CNN) is built by stacking Convolutional layers (to extract visual features like leaf textures, spots, and discolorations), MaxPooling layers (to condense information), and Dense layers (to perform the final classification).

```python
from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(256, 256, 3), num_classes=39):
    model = models.Sequential([
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flattening and Dense Layers for Classification
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Prevent overfitting
        layers.Dense(num_classes, activation='softmax') # 39 specific classes
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the model
cnn_model = create_cnn_model()
cnn_model.summary()
```

## 5. Classification of Leaf Type and Disease
The objective is to classify both the **leaf type** (e.g., Apple, Corn, Tomato) and the **disease type** (e.g., Apple Scab, Common Rust, Bacterial Spot). 
Our CNN model achieves this by utilizing the 39 distinct classes provided in the dataset. Because the dataset's folders are named in the format `[Plant_Species]___[Disease_Name]` (e.g., `Tomato___Early_blight`), predicting a single label effectively identifies both the plant species and its specific condition or disease simultaneously.

## 6. Training Data Split (80% / 20%)
To properly evaluate the model, we use an 80-20 split: 80% of the dataset is used for training the model, and the remaining 20% is held back for validating its accuracy to ensure it generalizes well to unseen data.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Point this to either the 'Original' or 'Augmentated/Plant_leave_diseases_dataset_with_augmentation' folder
DATASET_DIR = '../Original/'

# Image Data Generator with 20% Validation Split
datagen = ImageDataGenerator(
    rescale=1./255,      # Normalize pixel values
    validation_split=0.2 # Setting 20% split
)

# Training Dataset (80%)
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation Dataset (20%)
validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Training the model
# history = cnn_model.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=10
# )
```
