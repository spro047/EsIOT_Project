import os
import random
import shutil

# Paths are relative to the script location or absolute
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "DATASET", "Augmentated", "Plant_leave_diseases_dataset_with_augmentation")
backup_path = os.path.join(BASE_DIR, "DATASET", "Augmentated_Removed_Images")

target_classes = {
    "Orange___Haunglongbing_(Citrus_greening)": 2000,
    "Soybean___healthy": 2000,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 2000
}

def undersample(backup=True):
    print(f"Dataset path: {dataset_path}")
    if not os.path.exists(dataset_path):
        print("Error: Dataset path does not exist.")
        return

    if backup and not os.path.exists(backup_path):
        os.makedirs(backup_path)
        print(f"Created backup directory: {backup_path}")

    for class_name, target_count in target_classes.items():
        dir_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(dir_path):
            print(f"Skipping {class_name}, path not found: {dir_path}")
            continue
        
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        current_count = len(files)
        
        if current_count > target_count:
            num_to_remove = current_count - target_count
            print(f"Undersampling {class_name}: {current_count} -> {target_count} (moving {num_to_remove} to backup)")
            
            to_remove = random.sample(files, num_to_remove)
            
            if backup:
                class_backup = os.path.join(backup_path, class_name)
                if not os.path.exists(class_backup):
                    os.makedirs(class_backup)
                for f in to_remove:
                    shutil.move(os.path.join(dir_path, f), os.path.join(class_backup, f))
            else:
                for f in to_remove:
                    os.remove(os.path.join(dir_path, f))
        else:
            print(f"Class {class_name} already has {current_count} images or less. No action needed.")

    print("\nUndersampling complete.")

if __name__ == "__main__":
    undersample(backup=True)
