# 🌱 Soil Health Prediction System

This module implements a Machine Learning–based system to evaluate soil health using nutrient and environmental parameters. It is part of a larger plant health monitoring system integrating both soil and leaf analysis.

---

## 📌 Objective

To classify soil condition into:

* ✅ Good
* ⚠️ Moderate
* ❌ Poor

This helps in identifying nutrient deficiencies and supporting plant health diagnosis.

---

## 📊 Dataset

* Source: Crop Recommendation Dataset (Kaggle)
* Features used:

  * Nitrogen (N)
  * Phosphorus (P)
  * Potassium (K)
  * pH
  * Temperature
  * Humidity
  * Rainfall

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Loaded dataset using Pandas
* Created a new target variable: `soil_condition`
* Encoded categorical labels

---

### 2. Handling Class Imbalance

* Applied **SMOTE (Synthetic Minority Oversampling Technique)**
* Balanced dataset to ensure fair model learning

---

### 3. Model Used

* **Random Forest Classifier**

  * Handles nonlinear relationships
  * Suitable for tabular agricultural data
  * Provides feature importance insights

---

### 4. Training & Evaluation

* Train-Test Split: 80/20
* Evaluation Metrics:

  * Accuracy
  * Confusion Matrix
  * Precision, Recall, F1-score

---

## 📈 Results

* 🔹 Accuracy: ~98.5%
* 🔹 Balanced performance across all classes
* 🔹 High precision and recall

---

## 📊 Feature Importance

Top contributing features:

1. Potassium (K)
2. Nitrogen (N)
3. Phosphorus (P)
4. pH

This indicates that macronutrients play a crucial role in determining soil health.

---

## 🧠 Output

The model predicts:

👉 **Soil Condition: Good / Moderate / Poor**

---

## 🔗 Integration

This module is designed to work with a **leaf disease detection model (MobileNetV2)**:

* Leaf Model → Disease Detection
* Soil Model → Soil Condition
* Combined → Final Plant Health Diagnosis

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib / Seaborn

---

## 🚀 How to Run

1. Install dependencies:

```
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

2. Run the notebook:

```
jupyter notebook soil_model.ipynb
```

---

## 📌 Future Improvements

* Add real-time sensor data integration
* Improve soil condition labeling using domain datasets
* Deploy as a web application

---

## 👨‍💻 Contribution

This module focuses on soil health prediction and is part of a multi-modal plant health monitoring system.

---

## 📢 Summary

A Random Forest-based soil analysis system enhanced with SMOTE for class balancing, providing accurate and interpretable soil health predictions for smart agriculture applications.
