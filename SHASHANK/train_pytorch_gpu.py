import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler('cuda')

DATASET_DIR = '../DATASET/Augmentated/Plant_leave_diseases_dataset_with_augmentation'

class_counts = {}
for count_dir in os.listdir(DATASET_DIR):
    dir_path = os.path.join(DATASET_DIR, count_dir)
    if os.path.isdir(dir_path):
        class_counts[count_dir] = len(os.listdir(dir_path))

print(f'Total Classes: {len(class_counts)}')
print(f'Total Images: {sum(class_counts.values())}')

plt.figure(figsize=(15, 12))
sns.barplot(x=list(class_counts.values()), y=list(class_counts.keys()), palette='viridis', hue=list(class_counts.keys()), legend=False)
plt.title('Distribution of Images per Class')
plt.xlabel('Number of Images')
plt.ylabel('Class')
plt.tight_layout()
plt.savefig('eda_class_distribution.png')
plt.close()

IMAGE_SIZE = 224
BATCH_SIZE = 32

full_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=None
)

class_names = full_dataset.classes
num_classes = len(class_names)
print(f'Classes: {num_classes}')

targets = full_dataset.targets
train_idx, val_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.2,
    random_state=123,
    stratify=targets
)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

class FlexibleDataset(torch.utils.data.Dataset):
    def __init__(self, root, class_to_idx, filenames, labels, indices, transform=None):
        self.root = root
        self.class_to_idx = class_to_idx
        self.filenames = filenames
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = self.filenames[actual_idx]
        label = self.labels[actual_idx]
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

filenames = []
labels = []
for f in sorted(os.listdir(DATASET_DIR)):
    dir_path = os.path.join(DATASET_DIR, f)
    if os.path.isdir(dir_path):
        for img_name in os.listdir(dir_path):
            filenames.append(os.path.join(dir_path, img_name))
            labels.append(class_names.index(f))

train_dataset = FlexibleDataset(DATASET_DIR, full_dataset.class_to_idx, filenames, labels, train_idx, transform=train_transform)
val_dataset = FlexibleDataset(DATASET_DIR, full_dataset.class_to_idx, filenames, labels, val_idx, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

train_labels_full = [labels[i] for i in train_idx]
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels_full),
    y=train_labels_full
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_arr)}
class_weights_tensor = torch.tensor([class_weights_dict[i] for i in range(num_classes)], dtype=torch.float32).to(device)
print('Class weights computed.')

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rescale = lambda x: (x / 0.5) - 1.0
        self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.rescale(x)
        x = self.backbone.features(x)
        x = self.classifier(x)
        return x

model = MobileNetV2Classifier(num_classes).to(device)
print(f"Model initialized with {num_classes} output classes.")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

EPOCHS = 30
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 3

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("Starting GPU-accelerated training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_pytorch.pt')
        print('  New best model saved.')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f'Early stopping triggered after {epoch} epochs.')
            break

torch.save(model.state_dict(), 'cnn_mobilenet_pytorch_final.pt')
print(f'Model saved to cnn_mobilenet_pytorch_final.pt')

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(history['train_acc'], label='Train')
ax[0].plot(history['val_acc'], label='Val')
ax[0].set_title('Accuracy'); ax[0].legend()
ax[1].plot(history['train_loss'], label='Train')
ax[1].plot(history['val_loss'], label='Val')
ax[1].set_title('Loss'); ax[1].legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc='Evaluating'):
        inputs = inputs.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(24, 20))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Training complete. Plots saved.")
