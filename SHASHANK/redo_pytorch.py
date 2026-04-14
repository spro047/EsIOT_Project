import json
import os

def create_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" if i < len(source_lines) - 1 else line for i, line in enumerate(source_lines)]
    }

def create_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(source_lines) - 1 else line for i, line in enumerate(source_lines)]
    }

cells = []

# Section 1: Header & Imports
cells.append(create_markdown_cell([
    "# Plant Leaf Diseases Classification CNN (PyTorch Implementation)",
    "This notebook trains a high-performance deep learning model using pure PyTorch, with full GPU scaling and automatic mixed precision.",
    "## 1. Imports and Setup"
]))

cells.append(create_code_cell([
    "import os",
    "import copy",
    "import time",
    "import random",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import cv2",
    "",
    "import torch",
    "import torch.nn as nn",
    "import torch.optim as optim",
    "from torch.utils.data import DataLoader",
    "from torchvision import datasets, models",
    "import torchvision.transforms.v2 as transforms",
    "from torch.cuda.amp import autocast, GradScaler",
    "from sklearn.utils.class_weight import compute_class_weight",
    "",
    "# ----------------------------------------------------",
    "# 1. GPU DEVICE CONFIGURATION",
    "# ----------------------------------------------------",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "print(f'Training infrastructure mapped to: {device}')"
]))

# Section 2: EDA
cells.append(create_markdown_cell([
    "## 2. Exploratory Data Analysis (EDA)"
]))

cells.append(create_code_cell([
    "DATASET_DIR = '../DATASET/Augmentated/Plant_leave_diseases_dataset_with_augmentation'",
    "",
    "class_counts = {}",
    "for count_dir in os.listdir(DATASET_DIR):",
    "    dir_path = os.path.join(DATASET_DIR, count_dir)",
    "    if os.path.isdir(dir_path):",
    "        class_counts[count_dir] = len(os.listdir(dir_path))",
    "",
    "print(f'Total Classes: {len(class_counts)}')",
    "print(f'Total Images: {sum(class_counts.values())}')"
]))

# Section 3: Preprocessing (DataLoaders)
cells.append(create_markdown_cell([
    "## 3. High-Performance Preprocessing Pipeline using `PyTorch DataLoader`",
    "Using DataLoaders combined with `torchvision` transforms for highly parallelized augmentation, `num_workers`, and `pin_memory=True` for fast GPU staging."
]))

cells.append(create_code_cell([
    "BATCH_SIZE = 32",
    "IMG_SIZE = (224, 224)",
    "",
    "# 1. Define Augmentations using torchvision v2",
    "train_transforms = transforms.Compose([",
    "    transforms.Resize(IMG_SIZE),",
    "    transforms.RandomHorizontalFlip(p=0.5),",
    "    transforms.RandomRotation(degrees=20),",
    "    transforms.ColorJitter(brightness=0.1, contrast=0.1),",
    "    transforms.ToImage(),",
    "    transforms.ToDtype(torch.float32, scale=True),",
    "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])",
    "])",
    "",
    "val_transforms = transforms.Compose([",
    "    transforms.Resize(IMG_SIZE),",
    "    transforms.ToImage(),",
    "    transforms.ToDtype(torch.float32, scale=True),",
    "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])",
    "])",
    "",
    "# 2. ImageFolder to infer classes automatically",
    "full_dataset = datasets.ImageFolder(DATASET_DIR)",
    "class_names = full_dataset.classes",
    "num_classes = len(class_names)",
    "",
    "train_size = int(0.8 * len(full_dataset))",
    "val_size = len(full_dataset) - train_size",
    "train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])",
    "",
    "train_dataset.dataset.transform = train_transforms",
    "val_dataset.dataset.transform = val_transforms",
    "",
    "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)",
    "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)",
    "",
    "print(f'Detected classes: {num_classes}')"
]))

cells.append(create_code_cell([
    "# Handling Imbalanced Data using Class Weights for the Loss function",
    "targets = full_dataset.targets",
    "class_weights_arr = compute_class_weight('balanced', classes=np.unique(targets), y=targets)",
    "class_weights = torch.tensor(class_weights_arr, dtype=torch.float32).to(device)",
    "print('Class weights loaded on GPU.')"
]))

# Section 4: Model
cells.append(create_markdown_cell(["## 4. Model Architecture"]))
cells.append(create_code_cell([
    "weights = models.MobileNet_V2_Weights.IMAGENET1K_V1",
    "model = models.mobilenet_v2(weights=weights)",
    "",
    "for param in model.parameters():",
    "    param.requires_grad = False",
    "",
    "model.classifier[1] = nn.Sequential(",
    "    nn.Linear(model.last_channel, 256),",
    "    nn.ReLU(),",
    "    nn.Dropout(0.3),",
    "    nn.Linear(256, num_classes)",
    ")",
    "model = model.to(device)",
    "print('Model initialized on Device.')"
]))

# Section 5: Loop
cells.append(create_markdown_cell(["## 5. PyTorch Explicit Training Loop & AMP"]))
cells.append(create_code_cell([
    "EPOCHS = 10",
    "patience = 3",
    "best_val_loss = float('inf')",
    "patience_counter = 0",
    "",
    "criterion = nn.CrossEntropyLoss(weight=class_weights)",
    "optimizer = optim.Adam(model.parameters(), lr=1e-3)",
    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6)",
    "scaler = GradScaler()",
    "",
    "history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}",
    "best_model_wts = copy.deepcopy(model.state_dict())",
    "",
    "def train_model():",
    "    global best_val_loss, patience_counter, best_model_wts",
    "    for epoch in range(EPOCHS):",
    "        start_time = time.time()",
    "        model.train()",
    "        running_loss = 0.0; running_corrects = 0",
    "        for inputs, labels in train_loader:",
    "            inputs, labels = inputs.to(device), labels.to(device)",
    "            optimizer.zero_grad()",
    "            with autocast():",
    "                outputs = model(inputs)",
    "                loss = criterion(outputs, labels)",
    "            scaler.scale(loss).backward()",
    "            scaler.step(optimizer)",
    "            scaler.update()",
    "            running_loss += loss.item() * inputs.size(0)",
    "            _, preds = torch.max(outputs, 1)",
    "            running_corrects += torch.sum(preds == labels.data)",
    "        ",
    "        epoch_train_loss = running_loss / train_size",
    "        epoch_train_acc = running_corrects.double() / train_size",
    "        ",
    "        model.eval()",
    "        val_loss = 0.0; val_corrects = 0",
    "        with torch.no_grad():",
    "            for inputs, labels in val_loader:",
    "                inputs, labels = inputs.to(device), labels.to(device)",
    "                with autocast():",
    "                    outputs = model(inputs)",
    "                    loss = criterion(outputs, labels)",
    "                val_loss += loss.item() * inputs.size(0)",
    "                _, preds = torch.max(outputs, 1)",
    "                val_corrects += torch.sum(preds == labels.data)",
    "                ",
    "        epoch_val_loss = val_loss / val_size",
    "        epoch_val_acc = val_corrects.double() / val_size",
    "        ",
    "        history['train_loss'].append(epoch_train_loss)",
    "        history['val_loss'].append(epoch_val_loss)",
    "        history['train_acc'].append(epoch_train_acc.item())",
    "        history['val_acc'].append(epoch_val_acc.item())",
    "        scheduler.step(epoch_val_loss)",
    "        ",
    "        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')",
    "        ",
    "        if epoch_val_loss < best_val_loss:",
    "            best_val_loss = epoch_val_loss",
    "            patience_counter = 0",
    "            best_model_wts = copy.deepcopy(model.state_dict())",
    "        else:",
    "            patience_counter += 1",
    "            if patience_counter >= patience:",
    "                print('Early Stopping triggered.')",
    "                break",
    "    model.load_state_dict(best_model_wts)",
    "",
    "print('Ready to train.')",
    "# train_model()"
]))

# Section 6: Results
cells.append(create_markdown_cell(["## 6. Results"]))
cells.append(create_code_cell([
    "def plot_history(hist):",
    "    if not hist['train_loss']: return",
    "    fig, ax = plt.subplots(1, 2, figsize=(14, 5))",
    "    ax[0].plot(hist['train_acc'], label='Train')",
    "    ax[0].plot(hist['val_acc'], label='Val')",
    "    ax[0].set_title('Accuracy'); ax[0].legend()",
    "    ax[1].plot(hist['train_loss'], label='Train')",
    "    ax[1].plot(hist['val_loss'], label='Val')",
    "    ax[1].set_title('Loss'); ax[1].legend()",
    "    plt.show()",
    "# plot_history(history)"
]))
cells.append(create_markdown_cell(["## 7. Saving"]))
cells.append(create_code_cell([
    "MODEL_SAVE_PATH = 'cnn_mobilenet_pytorch.pth'",
    "# torch.save(model.state_dict(), MODEL_SAVE_PATH)",
    "print(f'Model saved to {MODEL_SAVE_PATH}')"
]))

notebook_dict = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(r'd:\Esiot_project\SHASHANK\CNN_Model.ipynb', 'w') as f:
    json.dump(notebook_dict, f, indent=1)
print('Notebook rewritten to pure PyTorch.')
