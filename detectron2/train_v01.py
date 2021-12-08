from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, launch, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# here we point our dataset json file and images folder
register_coco_instances("my_dataset_train1", {}, "/home/avena/blenderproc/datasets/stadard_01/coco_annotations.json", "/home/avena/blenderproc/datasets/stadard_01")
register_coco_instances("my_dataset_train3", {}, "/home/avena/blenderproc/datasets/standard3/coco_annotations.json", "/home/avena/blenderproc/datasets/standard3")
register_coco_instances("my_dataset_train4", {}, "/home/avena/blenderproc/datasets/standard4/coco_annotations.json", "/home/avena/blenderproc/datasets/standard4")
register_coco_instances("my_dataset_train5", {}, "/home/avena/blenderproc/datasets/standard5/coco_annotations.json", "/home/avena/blenderproc/datasets/standard5")
register_coco_instances("my_dataset_train6", {}, "/home/avena/blenderproc/datasets/standard6/coco_annotations.json", "/home/avena/blenderproc/datasets/standard6")
register_coco_instances("my_dataset_train7", {}, "/home/avena/blenderproc/datasets/standard7/coco_annotations.json", "/home/avena/blenderproc/datasets/standard7")


# if you have more than one data set you can register it here also, example:
# register_coco_instances("my_dataset_train_2", {}, "../datasets/stadard_02/coco_annotations.json", "../datasets/stadard_02")
# remember that name of dataset and both folders names (json and images) should be different from first one


from detectron2.engine import DefaultTrainer

# bitmask or polygon - polygons cannot have holes in mask, bitmask can

def main(args):
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.OUTPUT_DIR = "./output_R101FPN"


    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # extra dataset should be mentioned in line, after comma:
    cfg.DATASETS.TRAIN = ("my_dataset_train1", "my_dataset_train3",  "my_dataset_train5", "my_dataset_train6", "my_dataset_train7",)
    cfg.DATASETS.TEST = ()

    # how many threads should be used for data serialisation:
    cfg.DATALOADER.NUM_WORKERS = 4
    # if you train from scratch - comment out line bellow
    # if you want to continue training, use previous training weight file
    # cfg.MODEL.WEIGHTS = "/home/avena/PycharmProjects/pythonProject/output/model_final.pth"  # Let training initialize from previous training

    #how many images per batch - more is better but it is limited by gpu memory, but (???)
    # cfg.SOLVER.IMS_PER_BATCH = 6
    # learning rate, start with higher, later decrease
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR

    # number of bathes (epoch) - depend on how many pictures you have/batch size
    # cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # decrease learning rate in next  iterations
    # cfg.SOLVER.STEPS = []        # do not decay learning rate
    # how many the best regions are proposed for learinig
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)

    # number of clases !!!
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 26  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    #default output dir ./output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()

#name of weight files
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05   # set a custom testing threshold
#predictor = DefaultPredictor(cfg)


#from detectron2.utils.visualizer import ColorMode
#dataset_dicts_metadata = MetadataCatalog.get("my_dataset_train")
#dataset_dicts = DatasetCatalog.get("my_dataset_train")
#for d in random.sample(dataset_dicts, 5):
#    im = cv2.imread(d["file_name"])
#    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#    v = Visualizer(im[:, :, ::-1],
#                   metadata=dataset_dicts_metadata,
#                   scale=0.5,
#                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#    )
#    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#    cv2.imshow("", out.get_image()[:, :, ::-1])
#    cv2.waitKey(0)


# code for results calculation:
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# evaluator = COCOEvaluator("my_dataset_train", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "my_dataset_train")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    launch(
        main,
        2,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
