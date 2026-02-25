import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# -------- PATHS --------
MODEL_PATH = "models/resnet50_melanoma_final.keras"

TEST_DIR = "final_dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -------- LOAD MODEL --------
model = tf.keras.models.load_model(MODEL_PATH)

# -------- TEST DATA --------
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# -------- PREDICTIONS --------
y_true = test_gen.classes
y_pred_prob = model.predict(test_gen)
y_pred = (y_pred_prob > 0.3).astype(int).reshape(-1)

# -------- METRICS --------
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Benign", "Melanoma"]))

# -------- CONFUSION MATRIX --------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Melanoma"],
            yticklabels=["Benign", "Melanoma"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
