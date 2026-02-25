import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "resnet50_melanoma_final.keras"
IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Preprocess only once (IMPORTANT)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]
    prob_percent = round(float(prob) * 100, 2)
    prob_value = round(float(prob), 4)

    print("\n" + "="*60)
    print("MELANOMA DETECTION RESULT")
    print("="*60)
    print(f"Melanoma Probability: {prob_percent}% ({prob_value})")
    print("="*60)

    # ---- Risk Stratification ----
    if prob < 0.25:
        stage = 1
        risk = "✅ LOW RISK - LIKELY BENIGN"
        confidence = "Very Low Melanoma Risk"
        recommendation = "Routine monitoring recommended"
        visual = "🟢 GREEN"
    elif prob < 0.45:
        stage = 2
        risk = "⚠️ EARLY-STAGE MELANOMA SUSPICION"
        confidence = "Low-Moderate Melanoma Risk"
        recommendation = "Dermatologist consultation suggested"
        visual = "🟡 YELLOW"
    elif prob < 0.70:
        stage = 3
        risk = "⚠️⚠️ MODERATE RISK MELANOMA"
        confidence = "Moderate-High Melanoma Risk"
        recommendation = "Urgent dermatologist consultation needed"
        visual = "🟠 ORANGE"
    else:
        stage = 4
        risk = "🔴 HIGH RISK MELANOMA"
        confidence = "Very High Melanoma Risk"
        recommendation = "URGENT - Immediate specialist evaluation required"
        visual = "🔴 RED"

    print(f"\n📊 RISK STAGE: {stage}/4")
    print(f"Classification: {risk}")
    print(f"Confidence: {confidence}")
    print(f"Recommendation: {recommendation}")
    print(f"Status Indicator: {visual}")
    print("="*60 + "\n")

# ---- Run Prediction ----
predict_image("test.jpg")
