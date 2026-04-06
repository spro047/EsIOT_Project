# Plant Leaf Diseases Dataset Analysis

This document provides a comprehensive analysis of the "Plant Leaf Diseases Dataset," designed as foundational documentation for Deep Learning (DL) practitioners preparing to train classification or disease detection models.

## 1. Objective

The primary objective of this dataset is:
* **Foundation for DL Models:** To serve as a robust foundational dataset for developing and training Deep Learning models, such as Convolutional Neural Networks (CNNs).
* **Plant and Disease Identification:** The ultimate goal of these deep learning models is to accurately detect specific crop plant species and identify any diseases they may be carrying from visual leaf characteristics.

## 2. Dataset Analysis

### Folder Structure
The dataset follows a highly organized, hierarchical structure. At the root level, the dataset is split into two configurations:
- `Augmentated/`: Contains the augmented version of the dataset, inside a nested `Plant_leave_diseases_dataset_with_augmentation` directory.
- `Original/`: Contains the original, un-augmented version of the dataset.

Within each of these configurations, the images are directly organized into **39 distinct subdirectories**, where each subdirectory corresponds to a specific class label. The general naming convention of these subfolders is `[Plant_Species]___[Disease_Name]` or `[Plant_Species]___healthy`. There is also a special `Background_without_leaves` category.

Here is the complete folder structure of the dataset:

```text
Esiot_project/
├── Augmentated/
│   └── Plant_leave_diseases_dataset_with_augmentation/
│       ├── Apple___Apple_scab/
│       ├── Apple___Black_rot/
│       ├── Apple___Cedar_apple_rust/
│       ├── Apple___healthy/
│       ├── Background_without_leaves/
│       ├── Blueberry___healthy/
│       ├── Cherry___Powdery_mildew/
│       ├── Cherry___healthy/
│       ├── Corn___Cercospora_leaf_spot Gray_leaf_spot/
│       ├── Corn___Common_rust/
│       ├── Corn___Northern_Leaf_Blight/
│       ├── Corn___healthy/
│       ├── Grape___Black_rot/
│       ├── Grape___Esca_(Black_Measles)/
│       ├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
│       ├── Grape___healthy/
│       ├── Orange___Haunglongbing_(Citrus_greening)/
│       ├── Peach___Bacterial_spot/
│       ├── Peach___healthy/
│       ├── Pepper,_bell___Bacterial_spot/
│       ├── Pepper,_bell___healthy/
│       ├── Potato___Early_blight/
│       ├── Potato___Late_blight/
│       ├── Potato___healthy/
│       ├── Raspberry___healthy/
│       ├── Soybean___healthy/
│       ├── Squash___Powdery_mildew/
│       ├── Strawberry___Leaf_scorch/
│       ├── Strawberry___healthy/
│       ├── Tomato___Bacterial_spot/
│       ├── Tomato___Early_blight/
│       ├── Tomato___Late_blight/
│       ├── Tomato___Leaf_Mold/
│       ├── Tomato___Septoria_leaf_spot/
│       ├── Tomato___Spider_mites Two-spotted_spider_mite/
│       ├── Tomato___Target_Spot/
│       ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│       ├── Tomato___Tomato_mosaic_virus/
│       └── Tomato___healthy/
└── Original/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Apple___Cedar_apple_rust/
    ├── Apple___healthy/
    ├── Background_without_leaves/
    ├── Blueberry___healthy/
    ├── Cherry___Powdery_mildew/
    ├── Cherry___healthy/
    ├── Corn___Cercospora_leaf_spot Gray_leaf_spot/
    ├── Corn___Common_rust/
    ├── Corn___Northern_Leaf_Blight/
    ├── Corn___healthy/
    ├── Grape___Black_rot/
    ├── Grape___Esca_(Black_Measles)/
    ├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
    ├── Grape___healthy/
    ├── Orange___Haunglongbing_(Citrus_greening)/
    ├── Peach___Bacterial_spot/
    ├── Peach___healthy/
    ├── Pepper,_bell___Bacterial_spot/
    ├── Pepper,_bell___healthy/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    ├── Potato___healthy/
    ├── Raspberry___healthy/
    ├── Soybean___healthy/
    ├── Squash___Powdery_mildew/
    ├── Strawberry___Leaf_scorch/
    ├── Strawberry___healthy/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Early_blight/
    ├── Tomato___Late_blight/
    ├── Tomato___Leaf_Mold/
    ├── Tomato___Septoria_leaf_spot/
    ├── Tomato___Spider_mites Two-spotted_spider_mite/
    ├── Tomato___Target_Spot/
    ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
    ├── Tomato___Tomato_mosaic_virus/
    └── Tomato___healthy/
```

### File Analysis
* **File Types:** The dataset is composed exclusively of image files.
* **Image Formats:** The most predominant file extension across all folders is `.JPG`. Additionally, there are minor occurrences of `.jpg`, `.jpeg`, and a very small number of `.png` files. 
* **Metadata:** There are no explicit metadata files (such as CSV or JSON labels) included in the base directories. The class labels are solely determined implicitly based on the folder names holding the images.

### Image Content
* The images consist of close-up photographs of plant leaves. 
* Based on the `Background_without_leaves` folder, the dataset also contains non-leaf images to train models to distinguish between a leaf and a background environment. 
* Many images feature a single leaf positioned on a uniform background to emphasize the characteristics of the surface, though characteristics vary based on capturing conditions.

## 3. Dataset Information

### Leaf Information
The dataset captures the surface anatomy of individual leaves.
* Leaves range from fully healthy (vibrant and regular texture) to heavily diseased. 
* Features such as spots, molds, discolorations, blemishes, rusts, and blights are visually prominent on the surface of the diseased leaves.
* The focus is maintained on visual distinctness so that a CNN can effectively extract textural, shape, and color-based features.

### Disease Information
A wide variety of pathologies (viral, fungal, bacterial, and environmental) are represented. Key diseases include:
* **Fungal Infections:** Rusts (e.g., Common rust, Cedar apple rust), Scabs, Blights (Early/Late blight, Leaf blight), Mildew (Powdery mildew), and Molds (Leaf mold).
* **Bacterial Infections:** Bacterial spots and Haunglongbing (Citrus greening).
* **Viral Infections:** Mosaic virus and Yellow Leaf Curl Virus.
* **Pests/Mites:** Spider mites (Two-spotted spider mite).

### Plant and Disease Types (The 39 Classes)
The dataset covers **14 distinct crop species** plus a background class.

1. **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
2. **Blueberry:** Healthy
3. **Cherry:** Powdery Mildew, Healthy
4. **Corn (Maize):** Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
5. **Grape:** Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy
6. **Orange:** Haunglongbing (Citrus Greening)
7. **Peach:** Bacterial Spot, Healthy
8. **Pepper, Bell:** Bacterial Spot, Healthy
9. **Potato:** Early Blight, Late Blight, Healthy
10. **Raspberry:** Healthy
11. **Soybean:** Healthy
12. **Squash:** Powdery Mildew
13. **Strawberry:** Leaf Scorch, Healthy
14. **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-Spotted Spider Mite), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy
15. **Background:** Background without leaves

### Dataset Size
The dataset provides two variations which dictates its size:
* **Without Augmentation (Original):** Consists of roughly **~55,448 images** distributed across the 39 classes. These images are quite imbalanced; certain classes (e.g., Orange Citrus Greening) contain over 5,000 images, while others (e.g., Potato Healthy) contain as few as 152 images.
* **With Augmentation:** Standardizes and upsamples the dataset. Any class containing fewer than 1,000 images has been augmented (via rotations, flips, etc.) to reach a minimum of 1,000 images. Classes that already contained >= 1,000 images retain their original sizes. The overall dataset size increases to roughly **~61,000+ images**.

### Data Source/Origin
Based on the specific alignment of the 14 crop species and 38 disease/healthy pairings, this collection relies heavily on the **PlantVillage dataset**, a well-known open-access computer vision dataset built precisely for mobile disease diagnostics.

### Image Characteristics
* **Resolution & Lighting:** Typical of the PlantVillage source, images are largely square or standardized resolution (often 256x256), taken under fairly controlled or simulated lighting.
* **Color:** Full RGB color format representation, which is critical since disease markers distinctively alter the color profile of a leaf (e.g., yellowing, brown spotting).
* **Clarity:** Macro, close-up clarity is prioritized, isolating the leaf itself within the center of the frame and minimizing complex background noise where possible. 
