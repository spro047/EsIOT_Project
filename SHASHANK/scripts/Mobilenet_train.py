import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow version:", tf.__version__)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'DATASET', 'Augmentated', 'Plant_leave_diseases_dataset_with_augmentation')
train_ds = keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
bg_idx = class_names.index('Background_without_leaves')
filtered_class_names = [c for c in class_names if c != 'Background_without_leaves']
class_names = filtered_class_names
NUM_CLASSES = len(class_names)
print(f"Total classes: {len(class_names)} (removed Background_without_leaves)")
print(f"Training batches: {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}")

AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomContrast(0.2),
])

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Filter out Background_without_leaves and remap labels
def filter_background(images, labels):
    mask = labels != bg_idx
    images = tf.boolean_mask(images, mask)
    labels = tf.boolean_mask(labels, mask)
    labels = tf.where(labels > bg_idx, labels - 1, labels)
    return images, labels

train_ds = train_ds.map(filter_background, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(filter_background, num_parallel_calls=AUTOTUNE)

all_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
class_weight_dict = {i: class_weights[i] for i in range(NUM_CLASSES)}
print('Class weights computed.')

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_mobilenet_tf_unfitted.keras')
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
    print(f"Loaded converted TF model from {model_path}")
else:
    print(f"{model_path} not found. Build from scratch? (run convert_weights.py first)")

RETRAIN = True  # Set to True to fine-tune

if RETRAIN:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_mobilenet_tf_best.keras'), save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_mobilenet_tf_final.keras'))
    print('Retrained model saved.')

def evaluate_model(model, val_ds, class_names):
    all_preds, all_labels = [], []
    for images, labels in val_ds:
        preds = tf.argmax(model.predict(images, verbose=0), axis=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

evaluate_model(model, val_ds, class_names)


