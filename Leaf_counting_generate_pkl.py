import os
import cv2
import numpy as np
import pickle
import re

def load_images_from_folder(root_folder):
    image_filenames = []
    image_array = []
    metadata_list = []
    total_images = 0

    # Walk through directories in a sorted order
    for dirpath, _, filenames in sorted(os.walk(root_folder)):
        # Sort filenames to maintain order
        filenames = sorted(filenames, key=lambda x: x.lower())

        for file in filenames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(dirpath, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    image_filenames.append(file)
                    image_array.append(img)
                    metadata = extract_metadata_from_filename(file)
                    if metadata:
                        metadata_list.append(metadata)
                    total_images += 1
                    print(f"Processing {total_images}: {file}")

    return image_filenames, np.array(image_array, dtype=object), metadata_list

def extract_metadata_from_filename(filename):
    pattern = r"(\w+)_p(\d+)_d(\d+)_L(\d+)_(\d+)_(\d+)\.png"
    match = re.match(pattern, filename)
    if match:
        crop, plant, day, level, angle, leaf_count = match.groups()
        return {
            "crop": crop,
            "plant": int(plant),
            "day": int(day),
            "level": int(level),
            "angle": int(angle),
            "leaf_count": int(leaf_count),
        }
    return None

def save_pickle_data(pickle_file, image_filenames, image_array, metadata_list):
    with open(pickle_file, "wb") as f:
        pickle.dump((image_filenames, image_array, metadata_list), f)
    print(f"Data saved to {pickle_file}")

def load_pickle_data(pickle_file):
    with open(pickle_file, "rb") as f:
        image_filenames, image_array, metadata_list = pickle.load(f)
    print(f"Data loaded from {pickle_file}")
    return image_filenames, image_array, metadata_list

# Example usage
root_folder = "F:/Aman/Academics work/Awadh work/Grand challenge/ACM grand challenge/Recorded dataset for grand challenge/Crops data/Final_sorted_data/For_leaf_counting"
save_path = "F:/Aman/Academics work/Awadh work/Grand challenge/ACM grand challenge/Code/Pkl files/leaf_counting_data.pkl"

image_filenames, image_array, metadata_list = load_images_from_folder(root_folder)
save_pickle_data(save_path, image_filenames, image_array, metadata_list)

# Load and check the data
image_name, image_arr, meta_data = load_pickle_data(save_path)

print(image_name)
print(meta_data)
print(image_arr)

