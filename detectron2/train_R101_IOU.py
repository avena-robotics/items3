from detectron2.utils.logger import setup_logger
setup_logger()

import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import default_argument_parser, launch
# register_coco_instances("my_dataset_train5", {}, "/home/avena/blenderproc/datasets/standard/dataset1/coco_annotations.json", "/home/avena/blenderproc/datasets/standard/dataset1")
# register_coco_instances("my_dataset_train", {},  "/home/avena/blenderproc/datasets/standard/dataset2/coco_annotations.json", "/home/avena/blenderproc/datasets/standard/dataset2")
# register_coco_instances("my_dataset_train1", {}, "/home/avena/blenderproc/datasets/standard/dataset3/coco_annotations.json", "/home/avena/blenderproc/datasets/standard/dataset3")
# register_coco_instances("my_dataset_train2", {}, "/home/avena/blenderproc/datasets/random_domain/dataset1/coco_annotations.json", "/home/avena/blenderproc/datasets/random_domain/dataset1")
# register_coco_instances("my_dataset_train3", {}, "/home/avena/blenderproc/datasets/random_domain/dataset2/coco_annotations.json", "/home/avena/blenderproc/datasets/random_domain/dataset2")
# register_coco_instances("my_dataset_train4", {}, "/home/avena/blenderproc/datasets/random_domain/dataset3/coco_annotations.json", "/home/avena/blenderproc/datasets/random_domain/dataset3")
# register_coco_instances("my_dataset_train6", {}, "/home/avena/blenderproc/datasets/multiscenario_2/coco_annotations.json", "/home/avena/blenderproc/datasets/multiscenario_2")

# # TEXTURELESS
# register_coco_instances("my_dataset_train7", {}, "/home/avena/blenderproc/datasets/textureless1/coco_annotations.json", "/home/avena/blenderproc/datasets/textureless1")
# register_coco_instances("my_dataset_train8", {}, "/home/avena/blenderproc/datasets/textureless2/coco_annotations.json", "/home/avena/blenderproc/datasets/textureless2")
# register_coco_instances("my_dataset_train9", {}, "/home/avena/blenderproc/datasets/textureless3/coco_annotations.json", "/home/avena/blenderproc/datasets/textureless3")


# register_coco_instances("my_dataset_train1", {}, "/home/avena/datasets_new/datasets/dataset_001/coco_annotations.json", "/home/avena/datasets_new/datasets/dataset_001")
# register_coco_instances("my_dataset_train2", {}, "/home/avena/datasets_new/datasets/dataset_002/coco_annotations.json", "/home/avena/datasets_new/datasets/dataset_002")
# register_coco_instances("my_dataset_train3", {}, "/home/avena/datasets_new/datasets/dataset_003/coco_annotations.json", "/home/avena/datasets_new/datasets/dataset_003")
# register_coco_instances("my_dataset_train4", {}, "/home/avena/datasets_new/datasets/dataset_004/coco_annotations.json", "/home/avena/datasets_new/datasets/dataset_004")
# register_coco_instances("my_dataset_train5", {}, "/home/avena/datasets_new/datasets/dataset_005/coco_annotations.json", "/home/avena/datasets_new/datasets/dataset_005")
# register_coco_instances("my_dataset_val1", {}, "/home/avena/datasets_new/datasets/dataset_006/coco_annotations.json", "/home/avena/datasets_new/datasets/dataset_006")
register_coco_instances("dataset_train_1", {}, "/home/avena/multiscenario_2/coco_annotations.json", "/home/avena/multiscenario_2")
# register_coco_instances("dataset_train_2", {}, "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset1/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset1")
# register_coco_instances("dataset_train_3", {}, "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset2/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset2")
# register_coco_instances("dataset_train_4", {}, "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset3/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset3")
# register_coco_instances("dataset_train_5", {}, "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset4/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset4")
# register_coco_instances("dataset_train_6", {}, "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset5/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_dust/random/dataset5")
# register_coco_instances("dataset_train_7", {}, "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_textureless/random/dataset1/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_textureless/random/dataset1")
# register_coco_instances("dataset_train_8", {}, "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset1/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset1")
# register_coco_instances("dataset_train_9", {}, "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset2/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset2")
# register_coco_instances("dataset_train_10", {}, "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset3/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset3")

# register_coco_instances("dataset_train_1", {}, "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset1/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset1")
# register_coco_instances("dataset_train_2", {}, "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset2/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/worktop_HDRI/random/dataset2")
# register_coco_instances("dataset_train_3", {}, "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_shadows/random/dataset1/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_shadows/random/dataset1")
# register_coco_instances("dataset_train_4", {}, "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_shadows/random/dataset2/coco_annotations.json", "/home/avena/blenderproc/paper_datasets/HDRI_worktoptexture_shadows/random/dataset2")



register_coco_instances("validation_dataset", {}, "/home/avena/darwin_dataset/new_coco_intnames.json", "/home/avena/darwin_dataset")



from detectron2.engine import DefaultTrainer

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np
import logging

from detectron2.projects import point_rend

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)





class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            1000,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks

def main(args):
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.INPUT.MASK_FORMAT = "bitmask"

    # cfg.merge_from_file("/home/avena/blenderproc/detectron2/detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco.yaml")
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("dataset_train_1",)#, "dataset_train_2", "dataset_train_3", "dataset_train_4",) #, "dataset_train_2", "dataset_train_3", "dataset_train_4",)
                          
    cfg.DATASETS.TEST = ("validation_dataset",)


    # import random
    # fruits_nuts_metadata = MetadataCatalog.get("my_dataset_val1")
    # dataset_dicts = DatasetCatalog.get("my_dataset_val1")
    # for d in random.sample(dataset_dicts, 10):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], scale=1)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow("", vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)

    # GN PART
    # cfg.MODEL.RESNETS.NORM = "GN"
    # cfg.MODEL.FPN.NORM = "GN"
    # cfg.MODEL.ROI_MASK_HEAD.NORM = "GN"
    # cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"
    # END OF GN PART

    # TIGHTEN IOU
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.2, 0.8]
    # cfg.MODEL.RPN.NMS_THRESH = 0.6

    # TRY 152 BACKBONE

    # cfg.MODEL.RESNETS.DEPTH = 152
    cfg.MODEL.BACKBONE.FREEZE_AT = 3

    # cfg.MODEL.RPN.SMOOTH_L1_BETA = 1.0
    # cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 1.0

    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = ""
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.OUTPUT_DIR = "../paper_models/ResNet101_IOU"
    cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
    cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = [12000, 17000]        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 29  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # cfg.MODEL.POINT_HEAD.NUM_CLASSES = 29
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.TEST.EVAL_PERIOD = 500
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)


    # from detectron2.utils.visualizer import ColorMode
    # dataset_dicts_metadata = MetadataCatalog.get("my_dataset_train1")
    # dataset_dicts = DatasetCatalog.get("my_dataset_train1")
    # for d in random.sample(dataset_dicts, 5):
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    #     v = Visualizer(im[:, :, ::-1],
    #                    metadata=dataset_dicts_metadata,
    #                    scale=0.5,
    #                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #     )
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imshow("", out.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    launch(
        main,
        # 2 - number of GPUs
        1,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# evaluator = COCOEvaluator("my_dataset_train", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "my_dataset_train")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))