import torch
import tensorflow as tf
import numpy as np
import os

NUM_CLASSES = 38
PT_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_mobilenet_pytorch_final.pt')
TF_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_mobilenet_tf.keras')

print("Loading PyTorch checkpoint...")
pt_ckpt = torch.load(PT_PATH, map_location='cpu', weights_only=True)

print("Building TF MobileNetV2 model...")
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation='relu', name='classifier_dense_1')(x)
x = tf.keras.layers.BatchNormalization(name='classifier_bn')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='classifier_dense_2')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

print("Transferring classifier weights from PyTorch to TensorFlow...")
dense1_kernel = pt_ckpt['base_model.classifier.1.weight'].numpy().T
dense1_bias = pt_ckpt['base_model.classifier.1.bias'].numpy()

bn_gamma = pt_ckpt['base_model.classifier.3.weight'].numpy()
bn_beta = pt_ckpt['base_model.classifier.3.bias'].numpy()
bn_mean = pt_ckpt['base_model.classifier.3.running_mean'].numpy()
bn_var = pt_ckpt['base_model.classifier.3.running_var'].numpy()

dense2_kernel = pt_ckpt['base_model.classifier.5.weight'].numpy().T
dense2_bias = pt_ckpt['base_model.classifier.5.bias'].numpy()

model.get_layer('classifier_dense_1').set_weights([dense1_kernel, dense1_bias])
model.get_layer('classifier_bn').set_weights([bn_gamma, bn_beta, bn_mean, bn_var])
model.get_layer('classifier_dense_2').set_weights([dense2_kernel, dense2_bias])

print(f"Saving TF model to {TF_PATH}...")
model.save(TF_PATH)
print("Done! TF model saved successfully.")
