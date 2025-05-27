import os
import cv2
import json
import numpy as np

# Directories
image_dir = '/home/bushra/bushra_dslia/CamVid/train'
label_dir = '/home/bushra/bushra_dslia/CamVid/train_labels'

# Define classes: RGB colors and assigned category IDs
# Convert RGB to BGR for OpenCV
class_info = {
    1: {'name': 'Car',               'color_bgr': (128,   0,  64)},  # RGB (64, 0, 128)
    2: {'name': 'Pedestrian',       'color_bgr': (  0,  64,  64)},  # RGB (64, 64, 0)
    3: {'name': 'Bicyclist',        'color_bgr': (192, 128,   0)},  # RGB (0, 128, 192)
    4: {'name': 'MotorcycleScooter','color_bgr': (192,   0, 192)},  # RGB (192, 0, 192)
    5: {'name': 'Truck_Bus',        'color_bgr': (192, 128, 192)},  # RGB (192, 128, 192)
    6: {'name': 'SUVPickupTruck',   'color_bgr': (192, 128,  64)},  # RGB (64, 128, 192)
    7: {'name': 'TrafficLight',     'color_bgr': ( 64,  64,   0)}   # RGB (0, 64, 64)
}

# Prepare COCO structure
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Build categories list
for cat_id, info in class_info.items():
    coco['categories'].append({
        "id": cat_id,
        "name": info['name'],
        "supercategory": "object"  # or appropriate supercategory
    })

annotation_id = 1
image_id = 1

# Iterate over images
for img_file in sorted(os.listdir(image_dir)):
    if not img_file.lower().endswith('.png'):
        continue
    # Construct corresponding label filename
    base = os.path.splitext(img_file)[0]               # e.g. '0001TP_009210'
    label_file = f"{base}_L.png"                       # add suffix
    label_path = os.path.join(label_dir, label_file)
    img_path = os.path.join(image_dir, img_file)

    if not os.path.exists(label_path):
        print(f"Warning: Label file not found for {img_file}, skipping.")
        continue

    # Read image to get size
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}, skipping.")
        continue
    height, width = img.shape[:2]

    # Add image entry
    coco['images'].append({
        "id": image_id,
        "file_name": img_file,
        "width": width,
        "height": height
    })

    # Read label mask
    label_img = cv2.imread(label_path)
    if label_img is None:
        print(f"Warning: Could not read label {label_path}, skipping.")
        image_id += 1
        continue

    # For each class, find contours in the mask
    for cat_id, info in class_info.items():
        color_bgr = np.array(info['color_bgr'], dtype=np.uint8)
        # Create a binary mask where the class pixels are white
        mask = cv2.inRange(label_img, color_bgr, color_bgr)
        # Find contours of the objects of this class
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # For each contour, compute bounding box and create annotation
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = int(w * h)
            if area == 0:
                continue  # skip degenerate boxes

            # Optionally, get the segmentation polygon
            segmentation = cnt.flatten().tolist()
            if len(segmentation) >= 6:
                seg = [segmentation]  # list of one polygon
            else:
                seg = []

            coco['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0,
                "segmentation": seg
            })
            annotation_id += 1

    image_id += 1

# Save to JSON file
output_path = os.path.join(image_dir, 'train_annotations.json')
with open(output_path, 'w') as f:
    json.dump(coco, f, indent=4)

print(f"COCO annotations saved to {output_path}")
