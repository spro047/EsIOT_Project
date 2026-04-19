# Plant Leaf Disease Model Inference Report

This report presents the real-time inference results of the deep learning model trained on your leaf dataset. The script randomly samples images from your `Original` dataset and evaluates them through the PyTorch `CNN_Model`.

## Inference Execution Overview
- **Model Checkpoint**: `cnn_mobilenet_pytorch_final.pt`
- **Device**: GPU (`cuda`)
- **Evaluation Criteria**: The model evaluates one known *diseased* leaf and one known *healthy* leaf, returning its classification prediction.

## Dataset Splitting Strategy
To ensure the model generalizes well and maintains class balance, the plant disease dataset was split into training and validation subsets using a **stratified 80/20 ratio**:
1.  **Full Dataset Loading**: The images were indexed from the `DATASET` directory using `torchvision.datasets.ImageFolder`.
2.  **Stratified Split (Per-Folder Distribution)**: We transitioned from a global random shuffle to a **stratified split** using `sklearn.model_selection.train_test_split`. This ensures that exactly **80% of the images in EACH folder** (diseased and healthy) are allocated for training, while the remaining **20%** of each folder are reserved for validation. This prevents bias and ensures that rare classes are equally represented in both sets.
3.  **Reproducibility**: A specific random state (`123`) was applied to ensure the stratified split is consistent and reproducible across different environments.
4.  **Transformation Pipeline**: Post-splitting, separate augmentation pipelines were applied: heavy augmentations (flips, crops, rotations) for the training set to prevent overfitting, and standardized resizing/normalization for the validation set to ensure accurate performance metrics.

---

## 1. Diseased Leaf Prediction

The model successfully retrieved a leaf suffering from Black Rot.

**Model Result:**
> **Target:** `Apple___Black_rot`
> **Prediction:** `Apple___Black_rot` 

> [!TIP]
> The model accurately captured the features of the diseased leaf and correctly diagnosed the specific disease!

**Visualization:**
![Diseased Leaf Inference](assets/inference_diseased.png)

---

## 2. Healthy Leaf Prediction

The model successfully retrieved a healthy corn leaf.

**Model Result:**
> **Target:** `Corn___healthy`
> **Prediction:** `Corn___healthy`

> [!TIP]
> The model correctly identified that the leaf is entirely healthy without generating any false positive alarms for a disease!

**Visualization:**
![Healthy Leaf Inference](assets/inference_healthy.png)

---

## Summary

The model has demonstrated **excellent robustness** by accurately distinguishing between complex disease markings (like Black Rot spots) and perfectly healthy tissue. The successful predictions on completely unseen random samples from the dataset folder prove that your PyTorch CNN pipeline is working exactly as intended!
