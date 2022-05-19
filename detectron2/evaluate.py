from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import inference_on_dataset, COCOEvaluator, DatasetEvaluators

import os
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config

def setup():
    cfg = get_cfg()
    add_backbone_config(cfg)
    cfg.merge_from_file("/home/avena/blenderproc/detectron2/config/efnet.yaml")
    return cfg

cfg = get_cfg()
# cfg = setup()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "/home/avena/blenderproc/paper_models/ResNet152"
cfg.MODEL.RESNETS.DEPTH = 152
# cfg.MODEL.RESNETS.NORM = "GN"
# cfg.MODEL.FPN.NORM = "GN"
# cfg.MODEL.ROI_MASK_HEAD.NORM = "GN"
# cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"
# cfg.MODEL.RPN.IOU_THRESHOLDS = [0.2, 0.8]
# cfg.MODEL.RPN.NMS_THRESH = 0.6
# cfg.MODEL.RPN.SMOOTH_L1_BETA = 1.0
# cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 1.0

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 29  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)

register_coco_instances("validation",{}, "/home/avena/blenderproc/datasets/darwin_dataset/new_coco_intnames.json", "/home/avena/blenderproc/datasets/darwin_dataset")

evaluator = COCOEvaluator("validation", output_dir=cfg.OUTPUT_DIR, tasks=['bbox','segm'])
val_loader = build_detection_test_loader(cfg, "validation")
print(inference_on_dataset(predictor.model, val_loader, evaluator))


