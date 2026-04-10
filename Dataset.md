# Plant Leaf Diseases Dataset

## 1. Overview and Objective
This dataset serves as a comprehensive collection for training Deep Learning (DL) models, specifically Convolutional Neural Networks (CNNs), in agricultural technology applications. The dataset focuses on identifying crop species and detecting specific diseases from images of plant leaves.

## 2. Dataset Structure

The dataset is partitioned into two major main folders located under `DATASET/`:
- **Original/**: The raw, unaugmented collection of leaf images.
- **Augmentated/**: Contains a modified version of the dataset where underrepresented classes have been balanced out using data augmentation techniques (rotations, flips, etc.) to ensure a minimum of 1,000 images per class. 

Inside each folder, images are divided into **39 distinct subdirectories**, representing 14 crop species and 38 disease/healthy pairings, plus an extra background class. 

### Labeling Scheme (Directory-Based)
The dataset utilizes a **directory-based labeling scheme**, where the name of the folder explicitly acts as the label for all the images contained within it. This format is natively supported by deep learning data loaders like Keras `ImageDataGenerator.flow_from_directory()` or PyTorch `torchvision.datasets.ImageFolder()`.

The labeling convention follows a strict, compound naming string:
- **Format:** `[Plant_Species]___[Disease_Name]`
- **Delimiter:** Three consecutive underscores (`___`) separate the species name from the condition.
- **Healthy Status:** If the plant is healthy, the condition suffix is strictly `healthy` (e.g., `Apple___healthy`).
- **Disease Status:** For diseased conditions, the specific disease name is appended (e.g., `Apple___Apple_scab`).

There are no separate CSV or JSON mapping files provided or required. The folder structure itself is the ground truth.

## 3. Dataset Composition & Statistics

The dataset format is entirely composed of image files, primarily in `.JPG` formats (with minor `.jpg`, `.jpeg`, and `.png` file presence). The labels are implicitly defined by the directory names. 

### Total Volume:
- **Total Original Images:** 55,448
- **Total Augmented Images:** 61,486

### Distribution of Original Images (Per Class):

* **Apple**
  * Apple Scab: 630
  * Black Rot: 621
  * Cedar Apple Rust: 275
  * Healthy: 1,645

* **Blueberry**
  * Healthy: 1,502

* **Cherry**
  * Powdery Mildew: 1,052
  * Healthy: 854

* **Corn (Maize)**
  * Cercospora Leaf Spot / Gray Leaf Spot: 513
  * Common Rust: 1,192
  * Northern Leaf Blight: 985
  * Healthy: 1,162

* **Grape**
  * Black Rot: 1,180
  * Esca (Black Measles): 1,383
  * Leaf Blight (Isariopsis Leaf Spot): 1,076
  * Healthy: 423

* **Orange**
  * Haunglongbing (Citrus Greening): 5,507

* **Peach**
  * Bacterial Spot: 2,297
  * Healthy: 360

* **Pepper, Bell**
  * Bacterial Spot: 997
  * Healthy: 1,478

* **Potato**
  * Early Blight: 1,000
  * Late Blight: 1,000
  * Healthy: 152

* **Raspberry**
  * Healthy: 371

* **Soybean**
  * Healthy: 5,090

* **Squash**
  * Powdery Mildew: 1,835

* **Strawberry**
  * Leaf Scorch: 1,109
  * Healthy: 456

* **Tomato**
  * Bacterial Spot: 2,127
  * Early Blight: 1,000
  * Late Blight: 1,909
  * Leaf Mold: 952
  * Septoria Leaf Spot: 1,771
  * Spider Mites (Two-Spotted Spider Mite): 1,676
  * Target Spot: 1,404
  * Tomato Yellow Leaf Curl Virus: 5,357
  * Tomato Mosaic Virus: 373
  * Healthy: 1,591

* **Background**
  * Background Without Leaves: 1,143

## 4. Image Characteristics

* **Environment:** Images are taken against uniform backgrounds. The `Background_without_leaves` category can be used to help the model distinguish leaf matter from background environments.
* **Lighting and Quality:** Most images represent macro, close-up photography of leaf surfaces. They are square-sized and captured in RGB color.
* **Feature Representation:** Diseased images highlight various pathological indications such as yellowing, spotting, mold, rust formations, and blights. These are important textural and color features utilized for CNN-based feature extraction.

## 5. Disease and Pathology Coverage

The models trained on this dataset can learn to identify abnormalities deriving from:
* **Fungal Infections:** Rusts (e.g., Common rust), Scabs, Blights (Early/Late blight, Leaf blight), Mildew (Powdery mildew), and Molds.
* **Bacterial Infections:** Bacterial spots and Haunglongbing (Citrus greening).
* **Viral Infections:** Mosaic viruses and Yellow Leaf Curl Virion.
* **Pests/Mites:** E.g., Spider mites.

## 6. Data Augmentation (The `Augmentated/` Folder)

### Why was Data Augmentation Done?
The original dataset has a problem called **class imbalance**. For example, there are over 5,000 images of "Orange Citrus Greening", but only 152 images of "Healthy Potatoes". If we train an AI model on this unbalanced data, it will naturally become biased towards predicting the larger classes and perform poorly on the smaller ones because it hasn't seen enough examples. 

Data augmentation was performed to fix this imbalance by artificially creating more images for the categories that didn't have enough.

### How Much Augmentation was Done?
A strict rule was applied: **Every single category must have at least 1,000 images.**
- If a category already had 1,000 or more images in the `Original/` folder, it was left as is. No new images were created for it.
- If a category had fewer than 1,000 images (like the Healthy Potatoes with 152 images), the existing images were manipulated to generate enough new ones until the total count reached exactly 1,000.

Overall, this process increased the total size of the dataset from **55,448** original images to **61,486** images in the augmented version.

### How was the Data Augmented? (In Simple Words)
The new images weren't entirely new photos gathered from the real world. Instead, they were artificially generated by taking existing images from that specific category and slightly altering them using computer code. The techniques used included:

- **Rotations:** Slightly spinning the original leaf image (e.g., turning it by 15, 30, or 90 degrees).
- **Flipping:** Creating a mirror image of the leaf, either horizontally or vertically.
- **Zooming:** Slightly zooming in or out on the leaf structure.
- **Shifting:** Moving the leaf slightly to the left, right, up, or down within the frame.

To a Neural Network, a flipped or rotated leaf acts as a brand new training example. This helps the AI learn the features of the disease better and become more robust, all without needing researchers to physically photograph thousands of extra leaves!
