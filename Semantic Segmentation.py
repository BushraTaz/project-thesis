import os
import random
import numpy as np
import torch
from PIL import Image
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from torchvision.transforms import InterpolationMode

# ---------------- Configurations ----------------
IMG_SIZE = (960, 720)
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
SEED = 42
MODEL_SAVE_PATH = "deeplabv3plus_8class1.pth"
VAL_PRED_DIR = "val_predictions"

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------- Selected Classes ----------------
selected_classes = [
    "Road", "Sidewalk", "LaneMkgsDriv", "LaneMkgsNonDriv"
]

class_colors = {
    "Animal": [64,128,64], "Archway": [192,0,128], "Bicyclist": [0,128,192], "Bridge": [0,128,64],
    "Building": [128,0,0], "Car": [64,0,128], "CartLuggagePram": [64,0,192], "Child": [192,128,64],
    "Column_Pole": [192,192,128], "Fence": [64,64,128], "LaneMkgsDriv": [128,0,192],
    "LaneMkgsNonDriv": [192,0,64], "Misc_Text": [128,128,64], "MotorcycleScooter": [192,0,192],
    "OtherMoving": [128,64,64], "ParkingBlock": [64,192,128], "Pedestrian": [64,64,0],
    "Road": [128,64,128], "RoadShoulder": [128,128,192], "Sidewalk": [0,0,192], "SignSymbol": [192,128,128],
    "Sky": [128,128,128], "SUVPickupTruck": [64,128,192], "TrafficCone": [0,0,64],
    "TrafficLight": [0,64,64], "Train": [192,64,128], "Tree": [128,128,0], "Truck_Bus": [192,128,192],
    "Tunnel": [64,0,64], "VegetationMisc": [192,192,0], "Void": [0,0,0], "Wall": [64,192,0]
}

filtered_class_colors = {cls: class_colors[cls] for cls in selected_classes}
filtered_class_colors["Void"] = [0, 0, 0]

color_to_index = {tuple(v): i+1 for i, v in enumerate(filtered_class_colors.values()) if v != [0, 0, 0]}
color_to_index[(0, 0, 0)] = 0  # Void as index 0
index_to_color = {i: rgb for rgb, i in color_to_index.items()}

# ---------------- Dataset Class ----------------
class CamVidFilteredDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=IMG_SIZE):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.img_size = img_size
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")

        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img = self.color_jitter(img)

        img = transforms.Resize(self.img_size, interpolation=InterpolationMode.BILINEAR)(img)
        mask = transforms.Resize(self.img_size, interpolation=InterpolationMode.NEAREST)(mask)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        mask_np = np.array(mask, dtype=np.uint8)
        label_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)

        for rgb, class_idx in color_to_index.items():
            match = np.all(mask_np == rgb, axis=-1)
            label_mask[match] = class_idx

        return img, torch.from_numpy(label_mask).long()

# ---------------- Paths ----------------
train_image_dir = '/home/bushra/bushra_dslia/CamVid/train'
train_mask_dir = '/home/bushra/bushra_dslia/CamVid/train_labels'
val_image_dir = '/home/bushra/bushra_dslia/CamVid/val'
val_mask_dir = '/home/bushra/bushra_dslia/CamVid/val_labels'

# ---------------- Dataloaders ----------------
def seg_collate(batch):
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)

train_dataset = CamVidFilteredDataset(train_image_dir, train_mask_dir)
val_dataset = CamVidFilteredDataset(val_image_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=seg_collate, drop_last=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=seg_collate, drop_last=False, num_workers=NUM_WORKERS)

# ---------------- Model Setup ----------------
model = smp.DeepLabV3Plus(
    encoder_name='mobilenet_v2',
    encoder_weights='imagenet',
    in_channels=3,
    classes=len(color_to_index),
    activation=None
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------- Metric Calculation ----------------
def accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total

def iou(preds, labels, num_classes):
    iou_per_class = []
    for c in range(num_classes):
        intersection = ((preds == c) & (labels == c)).sum().item()
        union = ((preds == c) | (labels == c)).sum().item()
        if union == 0:
            iou_per_class.append(float('nan'))  # To handle the case where the union is zero
        else:
            iou_per_class.append(intersection / union)
    return iou_per_class

def mean_iou(iou_per_class):
    iou_per_class = [iou for iou in iou_per_class if not np.isnan(iou)]
    return np.mean(iou_per_class) if iou_per_class else 0.0

# ---------------- Training Loop ----------------
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy for the current batch
        preds = torch.argmax(outputs, dim=1)
        batch_accuracy = accuracy(preds, masks)
        total_accuracy += batch_accuracy

    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)
    print(f"✅ Epoch [{epoch+1}] Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accuracy: {avg_train_accuracy:.4f}")

# ---------------- Save Model ----------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to '{MODEL_SAVE_PATH}'.")

# ---------------- Validation & Inference ----------------
os.makedirs(VAL_PRED_DIR, exist_ok=True)
model.eval()
val_loss = 0.0
all_ious = []

# List of trained classes' indices (e.g., "Road", "Sidewalk", "LaneMkgsDriv", "LaneMkgsNonDriv")
trained_class_indices = [color_to_index[tuple(v)] for k, v in filtered_class_colors.items() if k != "Void"]

with torch.no_grad():
    for i, (images, masks) in enumerate(val_loader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        val_loss += loss.item()

        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        # Compute IoU per class
        iou_per_class = iou(preds, masks.cpu().numpy(), num_classes=len(color_to_index))
        all_ious.extend(iou_per_class)

        # Apply masking: Keep untrained classes in original form
        for j in range(preds.shape[0]):
            # Create the prediction image
            pred_img = np.zeros((*preds[j].shape, 3), dtype=np.uint8)
            for class_idx, rgb in index_to_color.items():
                # If the predicted class is trained, update it
                if class_idx in trained_class_indices:
                    pred_img[preds[j] == class_idx] = rgb
                # If the predicted class is untrained, keep the original class (use mask)
                else:
                    pred_img[masks[j].cpu().numpy() == class_idx] = rgb

            # Save the final masked prediction
            out_path = os.path.join(VAL_PRED_DIR, f"val_pred_{i*BATCH_SIZE+j}.png")
            Image.fromarray(pred_img).save(out_path)

# Calculate mean IoU
mean_iou_score = mean_iou(all_ious)
print(f"🎯 Final Validation Loss: {val_loss / len(val_loader):.4f}")
print(f"🎯 Final Mean IoU: {mean_iou_score:.4f}")
print(f"📁 Predictions saved in: {VAL_PRED_DIR}")
