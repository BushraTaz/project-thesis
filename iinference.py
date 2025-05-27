import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# ------------------- Class Info -------------------
class_info = {
    1: {'name': 'Car',               'color_bgr': (128,   0,  64)},
    2: {'name': 'Pedestrian',       'color_bgr': (  0,  64,  64)},
    3: {'name': 'Bicyclist',        'color_bgr': (192, 128,   0)},
    4: {'name': 'MotorcycleScooter','color_bgr': (192,   0, 192)},
    5: {'name': 'Truck_Bus',        'color_bgr': (192, 128, 192)},
    6: {'name': 'SUVPickupTruck',   'color_bgr': (192, 128,  64)},
    7: {'name': 'TrafficLight',     'color_bgr': ( 64,  64,   0)}
}

semantic_colors = {
    0: (0, 0, 0),              # Background
    1: (0, 0, 192),            # Sidewalk
    2: (128, 64, 128),         # Road
    3: (128, 0, 192),          # LaneMkgsDriv
    4: (192, 0, 64)            # LaneMkgsNonDriv
}

# ------------------- Load Models -------------------
def load_detectron2_model():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "/home/bushra/bushra_dslia/camvidroad_output/faster_rcnn_camvid_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    return DefaultPredictor(cfg)

def load_semantic_model():
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
        activation=None
    )
    model.load_state_dict(torch.load("/home/bushra/deeplabv3plus_8class1.pth", map_location='cpu'))
    model.eval()
    return model

# ------------------- Preprocessing -------------------
def preprocess_for_semantic(image_pil):
    transform = transforms.Compose([
        transforms.Resize((720, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0)

def get_semantic_mask(model, image_pil, original_size):
    input_tensor = preprocess_for_semantic(image_pil)
    with torch.no_grad():
        output = model(input_tensor)
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    color_mask = np.zeros((720, 960, 3), dtype=np.uint8)
    for class_idx, color in semantic_colors.items():
        color_mask[pred == class_idx] = color

    return cv2.resize(color_mask, original_size)

def get_object_detections(predictor, image_cv2, original_size):
    resized = cv2.resize(image_cv2, (960, 720))
    outputs = predictor(resized)

    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    labels = outputs["instances"].pred_classes.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()

    scale_x = original_size[0] / 960
    scale_y = original_size[1] / 720
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    return boxes, labels, scores

def blend_semantic_on_image(original, semantic_mask, alpha=0.6):
    return cv2.addWeighted(semantic_mask, alpha, original, 1 - alpha, 0)

def overlay_bounding_boxes(image, boxes, labels, scores):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[i]
        score = scores[i]
        color = class_info.get(label + 1, {'color_bgr': (0, 255, 0)})['color_bgr']
        name = class_info.get(label + 1, {'name': 'Unknown'})['name']
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{name} ({score:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

# ------------------- Traffic Decision Logic -------------------
def is_on_road(box, semantic_mask, road_class=2):
    x1, y1, x2, y2 = map(int, box)
    h, w = semantic_mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = semantic_mask[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    mask_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    road_pixels = np.sum((mask_gray == semantic_colors[road_class][0]))
    return road_pixels / crop.size > 0.1

def make_traffic_decision(boxes, labels, semantic_mask):
    vehicle_classes = {1, 3, 4, 5, 6}
    pedestrian_class = 2
    vehicle_count = 0
    pedestrian_on_road = 0

    for box, label in zip(boxes, labels):
        class_id = label + 1
        if class_id == pedestrian_class and is_on_road(box, semantic_mask):
            pedestrian_on_road += 1
        elif class_id in vehicle_classes:
            vehicle_count += 1

    if pedestrian_on_road > 0:
        return "🔴 RED: Pedestrian on road"
    elif vehicle_count > 6:
        return "🟢 GREEN: Heavy vehicle presence"
    elif vehicle_count > 0:
        return "🟡 YELLOW: Light traffic"
    else:
        return "🟢 GREEN: No traffic"

# ------------------- Main Video Function -------------------
def run_video_inference(video_path, output_path):
    print("🔄 Loading models...")
    object_detector = load_detectron2_model()
    semantic_segmenter = load_semantic_model()

    print("📼 Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error opening video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"🖼️ Processing frame {frame_count}...")
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        semantic_mask = get_semantic_mask(semantic_segmenter, image_pil, (frame_width, frame_height))
        boxes, labels, scores = get_object_detections(object_detector, frame, (frame_width, frame_height))

        traffic_decision = make_traffic_decision(boxes, labels, semantic_mask)
        blended = blend_semantic_on_image(frame, semantic_mask, alpha=0.6)
        final_frame = overlay_bounding_boxes(blended, boxes, labels, scores)

        # Draw traffic decision banner
        cv2.putText(final_frame, traffic_decision, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)

        out.write(final_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Video processing complete. Output saved at: {output_path}")

# ------------------- Run Inference -------------------
if __name__ == "__main__":
    input_video = "/home/bushra/bushra_dslia/vecteezy_busy-traffic-on-the-highway_6434705.mp4"
    output_video = "/home/bushra/output_video2.mp4"
    run_video_inference(input_video, output_video)






    import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from collections import Counter

# ------------------- Class Info -------------------
class_info = {
    1: {'name': 'Car',               'color_bgr': (128,   0,  64)},
    2: {'name': 'Pedestrian',       'color_bgr': (  0,  64,  64)},
    3: {'name': 'Bicyclist',        'color_bgr': (192, 128,   0)},
    4: {'name': 'TrafficLight',     'color_bgr': ( 64,  64,   0)}
}

semantic_colors = {
    0: (0, 0, 0),              # Background
    1: (0, 0, 192),            # Sidewalk
    2: (128, 64, 128),         # Road
    3: (128, 0, 192),          # LaneMkgsDriv
    4: (192, 0, 64)            # LaneMkgsNonDriv
}

# ------------------- Load Models -------------------
def load_detectron2_model():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "/home/bushra/bushra_dslia/camvidroad_output1/faster_rcnn_camvid_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    return DefaultPredictor(cfg)

def load_semantic_model():
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
        activation=None
    )
    model.load_state_dict(torch.load("/home/bushra/deeplabv3plus_8class1.pth", map_location='cpu'))
    model.eval()
    return model

# ------------------- Processing Functions -------------------
def read_image(path):
    return Image.open(path).convert("RGB")

def preprocess_for_semantic(image_pil):
    transform = transforms.Compose([
        transforms.Resize((720, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0)

def get_semantic_mask(model, image_pil, original_size):
    input_tensor = preprocess_for_semantic(image_pil)
    with torch.no_grad():
        output = model(input_tensor)
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    color_mask = np.zeros((720, 960, 3), dtype=np.uint8)
    for class_idx, color in semantic_colors.items():
        color_mask[pred == class_idx] = color
    return cv2.resize(color_mask, original_size), cv2.resize(pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)

def get_object_detections(predictor, image_cv2, original_size):
    resized = cv2.resize(image_cv2, (960, 720))
    outputs = predictor(resized)
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    labels = outputs["instances"].pred_classes.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    scale_x = original_size[0] / 960
    scale_y = original_size[1] / 720
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes, labels, scores

def blend_semantic_on_image(original, semantic_mask, alpha=0.6):
    return cv2.addWeighted(semantic_mask, alpha, original, 1 - alpha, 0)

def overlay_bounding_boxes(image, boxes, labels, scores):
    counter = Counter()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[i]
        score = scores[i]
        color = class_info.get(label + 1, {'color_bgr': (0, 255, 0)})['color_bgr']
        name = class_info.get(label + 1, {'name': 'Unknown'})['name']
        counter[name] += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{name} ({score:.2f})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image, counter

def make_traffic_decision(object_counts):
    cars = object_counts.get("Car", 0) + object_counts.get("Truck_Bus", 0) + object_counts.get("SUVPickupTruck", 0)
    pedestrians = object_counts.get("Pedestrian", 0)
    if pedestrians >= 1:
        return "🔴 RED: Pedestrian crossing"
    elif cars >= 5:
        return "🟢 GREEN: Heavy vehicle presence"
    elif pedestrians >= 3:
        return "🟡 YELLOW: Prepare to stop"
    else:
        return "🟢 GREEN: Normal flow"

def overlay_decision(image, decision):
    cv2.putText(image, f"Decision: {decision}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return image

# ------------------- Run Inference -------------------
def run_combined_inference(input_path, output_path):
    print("🔄 Loading models...")
    object_detector = load_detectron2_model()
    semantic_segmenter = load_semantic_model()

    print("🖼️ Reading image...")
    image_pil = read_image(input_path)
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    original_size = (image_cv2.shape[1], image_cv2.shape[0])

    print("🧠 Semantic segmentation...")
    semantic_mask, semantic_classes = get_semantic_mask(semantic_segmenter, image_pil, original_size)

    print("🔍 Object detection...")
    boxes, labels, scores = get_object_detections(object_detector, image_cv2, original_size)

    print("🎨 Blending and drawing...")
    blended = blend_semantic_on_image(image_cv2, semantic_mask)
    blended, counts = overlay_bounding_boxes(blended, boxes, labels, scores)

    print(f"✅ Object counts: {counts}")
    print("🚦 Making traffic light decision...")
    decision = make_traffic_decision(counts)
    print(f"📢 Decision: {decision}")
    final_img = overlay_decision(blended, decision)

    print(f"💾 Saving output to: {output_path}")
    cv2.imwrite(output_path, final_img)

# ------------------- Example Run -------------------
input_image = "/home/bushra/bushra_dslia/2.jpg"
output_image = "/home/bushra/image_combined_traffic_ai1.png"
run_combined_inference(input_image, output_image)
