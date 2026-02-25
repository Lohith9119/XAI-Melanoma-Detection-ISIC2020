import os
import shutil
import random

# ------------- PATHS (CHANGE ONLY IF YOUR PATH IS DIFFERENT) -------------
base_path = r"C:\Users\lohit\OneDrive\Desktop\Melanoma_Project"

preprocessed_path = os.path.join(base_path, "preprocessed")
src_melanoma = os.path.join(preprocessed_path, "melanoma")
src_benign   = os.path.join(preprocessed_path, "benign")

# This is where the new split folders will be created
dataset_path = os.path.join(base_path, "final_dataset")
train_path   = os.path.join(dataset_path, "train")
val_path     = os.path.join(dataset_path, "val")
test_path    = os.path.join(dataset_path, "test")

# Create folders if they don't exist
for p in [train_path, val_path, test_path]:
    os.makedirs(os.path.join(p, "melanoma"), exist_ok=True)
    os.makedirs(os.path.join(p, "benign"), exist_ok=True)

def split_class(src_folder, class_name, train_ratio=0.7, val_ratio=0.15):
    # Take only image files (you can add more extensions if needed)
    files = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    train_files = files[:n_train]
    val_files   = files[n_train:n_train + n_val]
    test_files  = files[n_train + n_val:]

    # Helper to copy a list of files to a target folder
    def copy_files(file_list, target_root):
        for fname in file_list:
            src = os.path.join(src_folder, fname)
            dst = os.path.join(target_root, class_name, fname)
            shutil.copy(src, dst)

    # Copy into train/val/test
    copy_files(train_files, train_path)
    copy_files(val_files, val_path)
    copy_files(test_files, test_path)

    print(f"{class_name}: {n_train} train, {n_val} val, {n_test} test (total {n})")


# ---------------------- RUN SPLIT FOR BOTH CLASSES ----------------------
if __name__ == "__main__":
    print("Splitting melanoma images...")
    split_class(src_melanoma, "melanoma")

    print("Splitting benign images...")
    split_class(src_benign, "benign")

    print("Done splitting dataset!")
