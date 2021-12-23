from detectron2.utils.logger import setup_logger

import cv2


from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

setup_logger()
register_coco_instances("my_dataset_train5", {}, "/home/avena/blenderproc/datasets/random_domain/dataset1/coco_annotations.json", "/home/avena/blenderproc/datasets/random_domain/dataset1")
register_coco_instances("my_dataset_train", {}, "/home/avena/blenderproc/datasets/standard/dataset1/coco_annotations.json", "/home/avena/blenderproc/datasets/standard/dataset1")
register_coco_instances("my_dataset_train1", {}, "/home/avena/blenderproc/datasets/standard/dataset2/coco_annotations.json", "/home/avena/blenderproc/datasets/standard/dataset2")
register_coco_instances("my_dataset_train2", {}, "/home/avena/blenderproc/datasets/standard/dataset3/coco_annotations.json", "/home/avena/blenderproc/datasets/standard/dataset3")
register_coco_instances("my_dataset_train3", {}, "/home/avena/blenderproc/datasets/random_domain/dataset2/coco_annotations.json", "/home/avena/blenderproc/datasets/random_domain/dataset2")
register_coco_instances("my_dataset_train4", {}, "/home/avena/blenderproc/datasets/random_domain/dataset3/coco_annotations.json", "/home/avena/blenderproc/datasets/random_domain/dataset3")
register_coco_instances("vis", {}, "/home/avena/blenderproc/datasets/standard_001_blender3/coco_annotations.json", "/home/avena/blenderproc/datasets/standard_001_blender3")




import random
fruits_nuts_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")
for d in random.sample(dataset_dicts, 293):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], scale=1)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)



