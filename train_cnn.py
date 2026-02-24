import tensorflow as tf
from tensorflow.keras import layers, models

BASE = r"C:\Users\lohit\OneDrive\Desktop\Melanoma_Project\final_dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    BASE + "/train", image_size=(224, 224), batch_size=32, label_mode="binary"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    BASE + "/val", image_size=(224, 224), batch_size=32, label_mode="binary"
)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=5)

model.save("cnn_melanoma.h5")

print("CNN Training Completed and Model Saved!")
