# 🌿 EsIOT Project: AI-Powered Smart Agriculture

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MobileNetV2](https://img.shields.io/badge/Architecture-MobileNetV2-blue)](https://arxiv.org/abs/1801.04381)
[![SmartAgri](https://img.shields.io/badge/Domain-Smart%20Agriculture-green)](#)

Welcome to the **EsIOT Project**, a comprehensive AI framework designed to enhance agricultural productivity through advanced plant disease classification and environment-aware crop recommendation systems.

---

## 🚀 Project Overview

This project focuses on providing end-to-end solutions for modern farming challenges:
1.  **Plant Leaf Disease Detection**: Leveraging deep learning to identify 39 different classes of diseases across various crops using MobileNetV2.
2.  **Soil Health & Recommendation**: utilizing historical soil data to predict the best crops for cultivation based on nutrient levels and environmental factors.

---

## 🛠 Features

### 🔍 Computer Vision Module (`SHASHANK/`)
-   **Architecture**: High-performance implementation of **MobileNetV2** tailored for plant disease classification.
-   **Dataset Management**: Advanced undersampling techniques applied to balanced the **Augmented Dataset**, reducing bias in over-represented classes (like Orange Greening, Soybean Healthy, and Tomato Yellow Leaf Curl Virus) to a standard 2000 images.
-   **EDA & Visualization**: Automated frameworks for analyzing class distribution, color intensities, and augmentation previews (`Dataset_Plots.ipynb`).
-   **Robust Training**: Built with PyTorch utilizing `torch.amp` for mixed-precision training and high GPU utilization.
-   **Inference Reports**: Ready-to-use checkpoints (`.pt`) with real-time prediction capabilities.

### 🧪 Soil & Crop Module (`ANUSHRI/`)
-   **Crop Recommendation**: Predictive modeling to suggest crops based on Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, and pH levels.
-   **Soil Health Pipeline**: Automated data processing and analysis of agricultural variables.

---

## 📂 Project Structure

```bash
EsIOT_Project/
├── SHASHANK/             # Computer Vision & Deep Learning
│   ├── Mobilenet.ipynb    # Training Pipeline (PyTorch)
│   ├── Dataset_Plots.ipynb# EDA & Data Visualization
│   ├── models.md          # Architecture details
│   └── assets/            # Prediction & Plot assets
├── ANUSHRI/              # Soil Health & Recommendation
│   ├── soil_health_pipeline.ipynb
│   └── Crop_recommendation.csv
├── DATASET/              # Image & Structured Data
│   ├── Augmentated/       # Processed leaf dataset (Balanced)
│   └── Original/          # Raw images
└── Papers/               # Reference Research Materials
```

---

## ⚙️ Preparation & Usage

### 1. Dataset Balancing
To ensure model fairness, we use a custom undersampling script to cap large classes at 2000 images:
```bash
python undersample_dataset.py
```

### 2. Training
Open `SHASHANK/Mobilenet.ipynb` to initiate the training loop. It automatically handles:
-   Image resizing & normalization (224x224).
-   Advanced augmentations (Rotation, Flips, Color Jitter).
-   Validation splits and Metric logging (Accuracy, F1-Score).

### 3. Inference
The model can be loaded via:
```python
model = models.mobilenet_v2(pretrained=False)
model.load_state_dict(torch.load('cnn_mobilenet_pytorch_final.pt'))
model.eval()
```

---

## 📊 Results Summary
The MobileNetV2 architecture demonstrates excellent robustness, effectively distinguishing between complex disease patterns and healthy leaf tissue even in low-contrast environments. Recent reports confirm high diagnostic accuracy across 39 distinct crop categories.

---

## 🤝 Contributors
-   **Shashank**: Computer Vision Specialist
-   **Anushri**: Data Scientist (Soil Analysis)
-   **Srushti**: Research & Dataset Curation
-   **Tanvi**: IoT Integration

---
*Developed for the Future of Sustainable Agriculture.*
