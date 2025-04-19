import os
import shutil
import random

# Paths
original_dataset_dir = "D:/Tumor Detection Model/dataset"
output_base_dir = "D:/Tumor Detection Model/dataset/split_dataset"

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Ensure output folders exist
classes = ["tumor", "normal"]
splits = ["train", "val", "test"]

for split in splits:
    for cls in classes:
        path = os.path.join(output_base_dir, split, cls)
        os.makedirs(path, exist_ok=True)

# Function to split and copy files
def split_and_copy(class_name):
    src_folder = os.path.join(original_dataset_dir, class_name)
    all_files = os.listdir(src_folder)
    random.shuffle(all_files)

    total = len(all_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    for file in train_files:
        shutil.copy(os.path.join(src_folder, file), os.path.join(output_base_dir, "train", class_name, file))

    for file in val_files:
        shutil.copy(os.path.join(src_folder, file), os.path.join(output_base_dir, "val", class_name, file))

    for file in test_files:
        shutil.copy(os.path.join(src_folder, file), os.path.join(output_base_dir, "test", class_name, file))

# Run split for each class
for cls in classes:
    split_and_copy(cls)

print("✅ Dataset split complete.")
