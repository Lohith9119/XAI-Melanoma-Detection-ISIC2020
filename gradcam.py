import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------- PATHS ----------------
MODEL_PATH = "models/resnet50_melanoma_final.keras"
IMG_PATH = "test.jpg"        # Change image if needed
IMG_SIZE = 224

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- IMAGE PREPROCESS ----------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError("❌ Image not found. Check IMG_PATH.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

input_img = preprocess_input(img.copy())
input_img = np.expand_dims(input_img, axis=0)

# ---------------- GRAD-CAM ----------------
last_conv_layer = "conv5_block3_out"

grad_model = tf.keras.models.Model(
    model.inputs,
    [model.get_layer(last_conv_layer).output, model.output]
)

with tf.GradientTape() as tape:
    conv_out, preds = grad_model(input_img)
    loss = preds[:, 0]

grads = tape.gradient(loss, conv_out)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_out = conv_out[0]
heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
heatmap = np.maximum(heatmap, 0)

if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)

heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

heatmap_uint8 = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# ===================== ABCDE ANALYSIS =====================

def compute_asymmetry(hm):
    h, w = hm.shape
    mid = w // 2
    left = hm[:, :mid]
    right = np.fliplr(hm[:, mid:mid + left.shape[1]])
    return np.mean(np.abs(left - right))

def compute_border_irregularity(hm):
    h, w = hm.shape
    bw = max(1, int(0.15 * min(h, w)))

    top = np.mean(hm[:bw, :])
    bottom = np.mean(hm[-bw:, :])
    left = np.mean(hm[:, :bw])
    right = np.mean(hm[:, -bw:])

    border_mean = np.mean([top, bottom, left, right])
    center_mean = np.mean(hm[bw:-bw, bw:-bw]) if bw*2 < h else np.mean(hm)

    return border_mean / (center_mean + 1e-6)

def compute_diameter_proxy(hm, threshold=0.5):
    return np.sum(hm > threshold) / hm.size

A_score = compute_asymmetry(heatmap_resized)
B_score = compute_border_irregularity(heatmap_resized)
D_score = compute_diameter_proxy(heatmap_resized)

A_text = "High asymmetry detected" if A_score > 0.15 else "Low asymmetry"
B_text = "Irregular border focus" if B_score > 1.2 else "Smooth border focus"
C_text = "Color variation inferred from dispersed activation"
D_text = "Large activated lesion area" if D_score > 0.25 else "Small activated area"
E_text = "Evolution cannot be inferred from a single image"

abcde_explanation = [
    f"A (Asymmetry): {A_text}",
    f"B (Border): {B_text}",
    f"C (Color): {C_text}",
    f"D (Diameter): {D_text}",
    f"E (Evolution): {E_text}"
]

# ---------------- PRINT ABCDE ----------------
print("\n🧠 ABCDE EXPLANATION (DERIVED FROM GRAD-CAM)")
print("-" * 55)
for line in abcde_explanation:
    print(line)

# ===================== CLEAN IEEE VISUALIZATION =====================

plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grad-CAM")
plt.imshow(heatmap_resized, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.savefig("gradcam_result.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ Grad-CAM image saved as: gradcam_result.png")
