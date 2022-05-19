import torch
import torchvision

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def build_model(num_classes=29):
    # backbone = torchvision.models.efficientnet_b7(pretrained=True).features
    backbone = torchvision.models.mobilenet_v2(pretrainefrom detectron2.engine import DefaultTrainer

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

from detectron2.projects import point_rendd=True).features
    # backbone.out_channels = 2560
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    model = MaskRCNN(backbone,
                     num_classes=2,
                     image_mean=image_mean,
                     image_std=image_std,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     mask_roi_pool=mask_roi_pooler)
    return model

model = build_model()
model.eval()
x = [torch.rand(3, 300, 400)]
predictions = model(x)
print(predictions[0]['scores'])
# pred = backbone(x)
# print(pred.shape)

transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(1000),
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.CocoDetection(root='/home/avena/blenderproc/datasets/dataset_001/',
                                             annFile='/home/avena/blenderproc/datasets/dataset_001/coco_annotations.json',
                                             transform=transform)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=lambda x: x)

# this line
# train_features, train_labels = next(iter(train_dataloader))
print(len(next(iter(train_dataloader))))
# or this lines
for i in range(5):
    print(i)
    j = 0
    for t, l in train_dataloader:
        print(j)
        j += 1


# print('Number of samples: ', len(cap))
# img, target = cap[3] # load 4th sample

# print("Image Size: ", img.size())
# print(target)
                                      