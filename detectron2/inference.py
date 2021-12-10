import json

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode


import argparse
import cv2
import numpy as np
import os

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("Weights", metavar="weights", type=str)
parser.add_argument("Path", metavar="path", type=str)
parser.add_argument("Threshold", metavar="thresh", type=float)
args = parser.parse_args()

# Catching parsed arguments
path = args.Path
threshold = args.Threshold
weights = args.Weights

# Setting model
cfg = get_cfg()

# Load config used for training
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

# Set number of classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 26

# Set weights (From arguments)
cfg.MODEL.WEIGHTS = weights

# Set threshold (From arguments)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set a custom testing threshold

# Create predictor
predictor = DefaultPredictor(cfg)

# Load image (From arguments)
im = cv2.imread(path)

# PREDICTION
outputs = predictor(im)

# Color visualization
v = Visualizer(im[:, :, ::-1],
                   # metadata=dataset_dicts_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
    )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Saving color
cv2.imwrite("inference_output/color_output.png", out.get_image()[:, :, ::-1])

# Getting data from inference

# Bounding boxes
bounding_boxes = outputs["instances"].pred_boxes.tensor.detach().to('cpu').numpy().tolist()

# Confidence scores
scores = outputs["instances"].scores.detach().to('cpu').numpy().tolist()

# Predicted classes (ids)
pred_classes = outputs["instances"].pred_classes.detach().to('cpu').numpy().tolist()

# Predicted masks (binary maps)
pred_masks = outputs["instances"].pred_masks.detach().to('cpu').numpy().astype(np.uint8)

# Make place for storing paths to masks
masks_paths = []

# Save each mask and store its path
for i, mask in enumerate(pred_masks):
    mask *= 255
    cv2.imwrite(f"inference_output/mask_{i}.png", mask)
    masks_paths.append(os.path.join(os.getcwd(), f"inference_output/mask_{i}.png"))

# Define dictionary of inference. This dictionary will be stored in json format
inference_results_dict = {"bounding_boxes": bounding_boxes, "pred_classes": pred_classes, "scores": scores,
                          "pred_masks": masks_paths}

# Save json
with open("inference_output/inferece_results.json", "w") as f:
    json.dump(inference_results_dict, f, indent=2)
