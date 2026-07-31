import os, sys, time, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
DATASET_DIR = os.path.abspath('DATASET/Augmentated/Plant_leave_diseases_dataset_with_augmentation')
PT_CKPT_PATH = os.path.join('SHASHANK', 'models', 'cnn_mobilenet_pytorch_final.pt')
TF_MODEL_PATH = os.path.join('SHASHANK', 'models', 'cnn_mobilenet_tf.keras')
EPOCHS = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)", flush=True)

train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading dataset...", flush=True)
full_dataset = torchvision.datasets.ImageFolder(root=DATASET_DIR)
class_names = full_dataset.classes
print(f"Found {len(full_dataset)} files belonging to {len(class_names)} classes.", flush=True)

# Filter out Background_without_leaves
bg_idx = class_names.index('Background_without_leaves')
filtered_class_names = [c for c in class_names if c != 'Background_without_leaves']
original_39 = class_names[:]
class_names = filtered_class_names
NUM_CLASSES = len(class_names)
print(f"Removed 'Background_without_leaves'. Using {NUM_CLASSES} classes.", flush=True)

# Label remap: old_label -> new_label (0..NUM_CLASSES-1)
old_to_new = {}
new_i = 0
for old_i, c in enumerate(original_39):
    if c != 'Background_without_leaves':
        old_to_new[old_i] = new_i
        new_i += 1

class FilteredLeafDataset(Dataset):
    def __init__(self, base_dataset, indices, old_to_new, transform):
        self.base = base_dataset
        self.indices = indices
        self.old_to_new = old_to_new
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, old_label = self.base[self.indices[idx]]
        return self.transform(img), self.old_to_new[old_label]

    @property
    def targets(self):
        return [self.old_to_new[self.base.targets[i]] for i in self.indices]

# Get valid sample indices (excluding background)
valid_indices = [i for i, t in enumerate(full_dataset.targets) if t != bg_idx]

# Split into train/val
import random
random.seed(123)
random.shuffle(valid_indices)
val_size = int(0.2 * len(valid_indices))
train_indices = valid_indices[val_size:]
val_indices = valid_indices[:val_size]

train_dataset = FilteredLeafDataset(full_dataset, train_indices, old_to_new, train_transform)
val_dataset = FilteredLeafDataset(full_dataset, val_indices, old_to_new, val_transform)

# Compute class weights with 3x tomato boost
train_labels = train_dataset.targets
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
class_weights = 1.0 / class_counts
tomato_new_indices = [i for i, c in enumerate(class_names) if c.startswith('Tomato___')]
TOMATO_BOOST = 3.0
for tc in tomato_new_indices:
    class_weights[tc] *= TOMATO_BOOST
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Tomato classes ({len(tomato_new_indices)}) boosted {TOMATO_BOOST}x", flush=True)

sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True
)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.base_model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.base_model.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)

model = PlantDiseaseModel()
print("Initialized model with pretrained ImageNet backbone.", flush=True)

model = model.to(device)

for param in model.base_model.features.parameters():
    param.requires_grad = False

params = model.base_model.classifier.parameters()
optimizer = optim.Adam(params, lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
)

print("Training on GPU...\n", flush=True)
best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]',
                unit='batch', leave=False, ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.*train_correct/train_total:.1f}%')

    avg_train_loss = train_loss / train_total
    train_acc = 100.0 * train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Val]',
                        unit='batch', leave=False, ncols=100)
        for images, labels in pbar_val:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            pbar_val.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.*val_correct/val_total:.1f}%')

    avg_val_loss = val_loss / val_total
    val_acc = 100.0 * val_correct / val_total
    epoch_time = time.time() - start_time

    print(f'Epoch {epoch:2d}/{EPOCHS} | {epoch_time:.0f}s | '
          f'Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | '
          f'Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%')

    scheduler.step(avg_val_loss)

    gap = train_acc - val_acc
    print(f'  Train-Val gap: {gap:.2f}%')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), PT_CKPT_PATH)
        print(f'  -> New best model saved (epoch {epoch})')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print(f'Early stopping after {epoch} epochs (val_loss did not improve for 5 epochs)')
            break
    if gap > 20.0:
        print(f'Overfitting detected (gap={gap:.1f}% > 20%), stopping early.')
        break

print(f'\nBest epoch: {best_epoch} (val_loss={best_val_loss:.4f})', flush=True)

model.load_state_dict(torch.load(PT_CKPT_PATH, map_location='cpu', weights_only=True))

print("Running weight conversion to TensorFlow...", flush=True)
result = subprocess.run(
    [sys.executable, os.path.join('SHASHANK', 'scripts', 'convert_weights.py')],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(f"convert_weights.py stderr:\n{result.stderr}", flush=True)

print("--- Evaluating on TF validation set ---", flush=True)
tf_model = keras.models.load_model(TF_MODEL_PATH)
print("TF model loaded.", flush=True)

val_ds_tf = keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset='validation',
    seed=123, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)

all_preds, all_true = [], []
for images, labels in val_ds_tf:
    mask = labels != bg_idx
    images_f = tf.boolean_mask(images, mask)
    labels_f = tf.boolean_mask(labels, mask)
    labels_remapped = tf.where(labels_f > bg_idx, labels_f - 1, labels_f)
    preds = tf.argmax(tf_model.predict(images_f, verbose=0), axis=1)
    all_preds.extend(preds.numpy())
    all_true.extend(labels_remapped.numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)
acc = np.mean(all_preds == all_true)
print(f"Validation Accuracy: {acc:.4f}", flush=True)
print(classification_report(all_true, all_preds, target_names=class_names, zero_division=0, digits=4), flush=True)

tf_model.save(os.path.join('SHASHANK', 'models', 'cnn_mobilenet_tf_final.keras'))
print("Final TF model saved: SHASHANK/models/cnn_mobilenet_tf_final.keras", flush=True)
print("\nDONE.", flush=True)
