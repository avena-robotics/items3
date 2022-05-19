import json

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.projects import point_rend


import argparse
import cv2
import numpy as np
import os

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("Weights", metavar="weights", type=str)
parser.add_argument("Path", metavar="path", type=str)
parser.add_argument("-t", '--threshold', default=0.5, type=float, help='threshold of confidence. predictions below'
                                                                       'threshold are rejected')
parser.add_argument('-s', '--scale', default=1, type=float, help='scale of the inference with respect to the input')
parser.add_argument('-m', '--model', default=0, type=float, help='0 - R50 backbone, 1 - R101 backbone', choices=[0, 1, 2])

args = parser.parse_args()

# Catching parsed arguments
path = args.Path
threshold = args.threshold
weights = args.Weights
# Setting model
cfg = get_cfg()

# Load config used for training
if args.model == 0:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
elif args.model == 1:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
# else:
#     raise ValueError("Model of this number is not supported yet")
elif args.model == 2:
    point_rend.add_pointrend_config(cfg)
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 29


cfg.MODEL.RPN.IOU_THRESHOLDS = [0.2, 0.8]
cfg.MODEL.RPN.NMS_THRESH = 0.6
cfg.MODEL.RPN.SMOOTH_L1_BETA = 1.0
cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 1.0

# Set number of classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 29
# cfg.MODEL.RESNETS.NORM = "GN"
# cfg.MODEL.FPN.NORM = "GN"
# cfg.MODEL.ROI_MASK_HEAD.NORM = "GN"
# cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"
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
                   scale=args.scale,
                   instance_mode=ColorMode.IMAGE_BW
    )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

output_folder = "ptcld4_output"

# Saving color
cv2.imwrite(f"{output_folder}/color_output.png", out.get_image()[:, :, ::-1])

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
    cv2.imwrite(f"{output_folder}/mask_{i}.png", mask)
    masks_paths.append(os.path.join(os.getcwd(), f"inference_output/mask_{i}.png"))

# Define dictionary of inference. This dictionary will be stored in json format
inference_results_dict = {"bounding_boxes": bounding_boxes, "pred_classes": pred_classes, "scores": scores,
                          "pred_masks": masks_paths}

# Save json
with open(f"{output_folder}/inferece_results.json", "w") as f:
    json.dump(inference_results_dict, f, indent=2)
