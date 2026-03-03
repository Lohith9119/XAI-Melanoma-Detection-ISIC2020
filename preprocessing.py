<<<<<<< HEAD
import os
import shutil
import pandas as pd
from PIL import Image

# ------------------------------------------
# STEP 1: Define Paths
# ------------------------------------------
base_path = r"C:\Users\lohit\OneDrive\Desktop\Melanoma_Project"


images_path = os.path.join(base_path, "ISIC_2020_Training_JPEG", "train")
csv_path = os.path.join(base_path, "ISIC_2020_Training_GroundTruth.csv")

dataset_path = os.path.join(base_path, "dataset")
melanoma_path = os.path.join(dataset_path, "melanoma")
benign_path = os.path.join(dataset_path, "benign")

preprocessed_path = os.path.join(base_path, "preprocessed")
prep_melanoma = os.path.join(preprocessed_path, "melanoma")
prep_benign = os.path.join(preprocessed_path, "benign")

# ------------------------------------------
# STEP 2: Create Required Folders
# ------------------------------------------
os.makedirs(melanoma_path, exist_ok=True)
os.makedirs(benign_path, exist_ok=True)
os.makedirs(prep_melanoma, exist_ok=True)
os.makedirs(prep_benign, exist_ok=True)

# ------------------------------------------
# STEP 3: Load CSV
# ------------------------------------------
df = pd.read_csv(csv_path)
print("CSV Loaded Successfully")
print(df.head())

# ------------------------------------------
# STEP 4: Sort Images into melanoma/benign
# ------------------------------------------
print("Sorting images...")

missing = 0

for idx, row in df.iterrows():
    img_id = row["image_name"]
    label = row["target"]
    filename = img_id + ".jpg"

    src = os.path.join(images_path, filename)

    if not os.path.exists(src):
        missing += 1
        continue

    if label == 1:
        dst = os.path.join(melanoma_path, filename)
    else:
        dst = os.path.join(benign_path, filename)

    shutil.copy(src, dst)

    if idx % 1000 == 0:
        print(f"Processed {idx} images...")

print("Sorting Completed")
print("Missing images:", missing)

# ------------------------------------------
# STEP 5: Preprocessing Function
# ------------------------------------------
def preprocess_folder(src_folder, dst_folder, size=(224,224)):
    files = os.listdir(src_folder)
    print(f"Found {len(files)} images in {src_folder}")

    count = 0
    for img_name in files:
        src = os.path.join(src_folder, img_name)
        dst = os.path.join(dst_folder, img_name)

        try:
            img = Image.open(src).convert("RGB")
            img = img.resize(size)
            img.save(dst)
        except Exception as e:
            print("Error:", img_name, e)
            continue

        count += 1
        if count % 1000 == 0:
            print(f"Preprocessed {count} images...")

    print(f"Total preprocessed from {src_folder}: {count}")

# ------------------------------------------
# STEP 6: Run Preprocessing
# ------------------------------------------
print("Starting preprocessing...")

preprocess_folder(melanoma_path, prep_melanoma)
preprocess_folder(benign_path, prep_benign)

print("PREPROCESSING COMPLETE!")
=======
import os
import shutil
import pandas as pd
from PIL import Image

# ------------------------------------------
# STEP 1: Define Paths
# ------------------------------------------
base_path = r"C:\Users\lohit\OneDrive\Desktop\Melanoma_Project"


images_path = os.path.join(base_path, "ISIC_2020_Training_JPEG", "train")
csv_path = os.path.join(base_path, "ISIC_2020_Training_GroundTruth.csv")

dataset_path = os.path.join(base_path, "dataset")
melanoma_path = os.path.join(dataset_path, "melanoma")
benign_path = os.path.join(dataset_path, "benign")

preprocessed_path = os.path.join(base_path, "preprocessed")
prep_melanoma = os.path.join(preprocessed_path, "melanoma")
prep_benign = os.path.join(preprocessed_path, "benign")

# ------------------------------------------
# STEP 2: Create Required Folders
# ------------------------------------------
os.makedirs(melanoma_path, exist_ok=True)
os.makedirs(benign_path, exist_ok=True)
os.makedirs(prep_melanoma, exist_ok=True)
os.makedirs(prep_benign, exist_ok=True)

# ------------------------------------------
# STEP 3: Load CSV
# ------------------------------------------
df = pd.read_csv(csv_path)
print("CSV Loaded Successfully")
print(df.head())

# ------------------------------------------
# STEP 4: Sort Images into melanoma/benign
# ------------------------------------------
print("Sorting images...")

missing = 0

for idx, row in df.iterrows():
    img_id = row["image_name"]
    label = row["target"]
    filename = img_id + ".jpg"

    src = os.path.join(images_path, filename)

    if not os.path.exists(src):
        missing += 1
        continue

    if label == 1:
        dst = os.path.join(melanoma_path, filename)
    else:
        dst = os.path.join(benign_path, filename)

    shutil.copy(src, dst)

    if idx % 1000 == 0:
        print(f"Processed {idx} images...")

print("Sorting Completed")
print("Missing images:", missing)

# ------------------------------------------
# STEP 5: Preprocessing Function
# ------------------------------------------
def preprocess_folder(src_folder, dst_folder, size=(224,224)):
    files = os.listdir(src_folder)
    print(f"Found {len(files)} images in {src_folder}")

    count = 0
    for img_name in files:
        src = os.path.join(src_folder, img_name)
        dst = os.path.join(dst_folder, img_name)

        try:
            img = Image.open(src).convert("RGB")
            img = img.resize(size)
            img.save(dst)
        except Exception as e:
            print("Error:", img_name, e)
            continue

        count += 1
        if count % 1000 == 0:
            print(f"Preprocessed {count} images...")

    print(f"Total preprocessed from {src_folder}: {count}")

# ------------------------------------------
# STEP 6: Run Preprocessing
# ------------------------------------------
print("Starting preprocessing...")

preprocess_folder(melanoma_path, prep_melanoma)
preprocess_folder(benign_path, prep_benign)

print("PREPROCESSING COMPLETE!")
>>>>>>> 7c563abf4a2d298be24af7279f8c1aabb25be1f3
