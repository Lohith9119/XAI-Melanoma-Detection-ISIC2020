# 🔬 Explainable AI Framework for Early Melanoma Detection

## 📌 Overview
This project implements a deep learning-based melanoma detection system using:

- Custom CNN
- ResNet50 (Transfer Learning)
- Grad-CAM for Explainability

Dataset: ISIC 2020 Skin Lesion Dataset

---

## 🧠 Key Features
- Binary classification: Benign vs Malignant
- Transfer learning using ResNet50
- Grad-CAM visualization for model interpretability
- Preprocessing and dataset splitting pipeline

---

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## 📊 Results
- ResNet50 achieved high classification accuracy
- Grad-CAM heatmaps correctly highlight lesion regions
- Explainability aligned with dermatology ABCDE rule

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train_resnet_final.py
python gradcam.py
