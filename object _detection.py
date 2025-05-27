import os
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# --- Paths ---
train_json = '/home/bushra/bushra_dslia/CamVid/train/train_annotations.json'
val_json = '/home/bushra/bushra_dslia/CamVid/val/val_annotations.json'  # add your val JSON here
train_image_root = '/home/bushra/bushra_dslia/CamVid/train/'
val_image_root = '/home/bushra/bushra_dslia/CamVid/val/'

train_dataset_name = "camvid_train"
val_dataset_name = "camvid_val"

# --- Register datasets ---
register_coco_instances(train_dataset_name, {}, train_json, train_image_root)
register_coco_instances(val_dataset_name, {}, val_json, val_image_root)

# --- Config Setup ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (train_dataset_name,)
cfg.DATASETS.TEST = (val_dataset_name,)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

# IMPORTANT: Change number of classes to your dataset (4 classes)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

# Hyperparameters and output
cfg.SOLVER.MAX_ITER = 5000
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.STEPS = (500, 800)
cfg.OUTPUT_DIR = "/home/bushra/bushra_dslia/camvidroad_output1"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Device (use "cuda" if available)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Trainer ---
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# --- Evaluation on Validation Set ---
evaluator = COCOEvaluator(val_dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, val_dataset_name)
metrics = inference_on_dataset(trainer.model, val_loader, evaluator)

print("===== Validation Metrics =====")
print(metrics)

# --- Save final weights ---
final_model_path = os.path.join(cfg.OUTPUT_DIR, "faster_rcnn_camvid_final.pth")
torch.save(trainer.model.state_dict(), final_model_path)
print(f"✅ Model saved to: {final_model_path}")
