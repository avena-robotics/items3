import detectron2 
from detectron2.config import get_cfg
from detectron2_backbone import backbone
from detectron2_backbone.config import add_backbone_config

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

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import default_argument_parser, launch

import os

def setup():
    cfg = get_cfg()
    add_backbone_config(cfg)
    cfg.merge_from_file("/home/avena/blenderproc/detectron2/config/efnet.yaml")
    return cfg

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
            500,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks


def main(args):
    cfg = setup()
    # print(cfg)

    # register_coco_instances("my_dataset_train1", {}, "/home/avena/blenderproc/datasets/dataset_001/coco_annotations.json", "/home/avena/blenderproc/datasets/dataset_001")
    # register_coco_instances("my_dataset_train2", {}, "/home/avena/blenderproc/datasets/dataset_002/coco_annotations.json", "/home/avena/blenderproc/datasets/dataset_002")
    # register_coco_instances("my_dataset_train3", {}, "/home/avena/blenderproc/datasets/dataset_003/coco_annotations.json", "/home/avena/blenderproc/datasets/dataset_003")
    # register_coco_instances("my_dataset_train4", {}, "/home/avena/blenderproc/datasets/dataset_004/coco_annotations.json", "/home/avena/blenderproc/datasets/dataset_004")
    # register_coco_instances("my_dataset_train5", {}, "/home/avena/blenderproc/datasets/dataset_005/coco_annotations.json", "/home/avena/blenderproc/datasets/dataset_005")
    # register_coco_instances("my_dataset_train6", {}, "/home/avena/blenderproc/datasets/dataset_006/coco_annotations.json", "/home/avena/blenderproc/datasets/dataset_006")
    # register_coco_instances("my_dataset_train7", {}, "/home/avena/blenderproc/datasets/multiscenario_2/coco_annotations.json", "/home/avena/blenderproc/datasets/multiscenario_2")

    register_coco_instances("my_dataset_train1", {}, "/home/avena/multiscenario_2/coco_annotations.json", "/home/avena/multiscenario_2")

    register_coco_instances("validation",{}, "/home/avena/blenderproc/datasets/darwin_dataset/new_coco_intnames.json", "/home/avena/blenderproc/datasets/darwin_dataset")
    # register_coco_instances("validation2",{}, "/home/avena/blenderproc/datasets/old_darwin_dataset/new_coco_intnames.json", "/home/avena/blenderproc/datasets/old_darwin_dataset")




    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.DATASETS.TRAIN = ("my_dataset_train1", )

    cfg.DATASETS.TEST = ("validation", )

    # cfg.MODEL.RESNETS.NORM = "GN"
    # cfg.MODEL.FPN.NORM = "GN"
    # cfg.MODEL.ROI_MASK_HEAD.NORM = "GN"
    # cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"


    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 29

    cfg.OUTPUT_DIR = "../paper_models/EfficientNet"
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025 # pick a good LR
    cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = [12000, 17000] 
    cfg.TEST.EVAL_PERIOD = 500

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)


    from detectron2.utils.visualizer import ColorMode
    import random
    import cv2

    # dataset_dicts_metadata = MetadataCatalog.get("validation")
    # dataset_dicts = DatasetCatalog.get("validation")
    # for d in random.sample(dataset_dicts, 30):
    #     im = cv2.imread(d["file_name"])
    #     print(d["file_name"])
    #     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    #     v = Visualizer(im[:, :, ::-1],
    #                     metadata=dataset_dicts_metadata,
    #                     scale=1,
    #                     instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
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