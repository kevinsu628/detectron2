import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import argparse

# import some common detectron2 utilities
#from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


import random
import os
import glob
from detectron2.structures import BoxMode


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 training whole folder")
    parser.add_argument("--dataset_folder", help="A folder of images to train")
    parser.add_argument("--output", help="A folder to store checkpoints ")
    parser.add_argument(
        "--config",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        help="path to config file"
    )
    parser.add_argument(
        "--weight",
        help="path to the pretrained weight"
    )
    parser.add_argument(
        "--dataset_name",
        help="The name of this registering dataset",
        default="ds"
    )
    parser.add_argument(
        "--ext",
        help="extention of the image",
        default="jpg"
    )
    return parser


# write a function that loads the dataset into detectron2's standard format
def get_dicts(img_dir, args):
    dataset_dicts = []
    for each_txt in glob.glob(os.path.join(img_dir, "*.txt")):
        record = {}

        filename = each_txt.replace(".txt", "."+args.ext)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width

        objs = []
        txt_reader = open(each_txt, "r")
        for row in txt_reader:
            row = row.strip("\n").split(" ")
            cls_id = row[0]
            x_c, y_c, w, h = [float(x) for x in row[1:]]
            x_0, y_0 = (x_c - w / 2) * width, (y_c - h / 2) * height
            x_1, y_1 = (x_c + w / 2) * width, (y_c + h / 2) * height
            obj = {
                "bbox": [x_0, y_0, x_1, y_1],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(cls_id)
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


args = get_parser().parse_args()

for d in ["train", "val"]:
    DatasetCatalog.register(os.path.join(args.dataset_name, d), lambda d=d: get_dicts(args.dataset_folder + d, args))
    MetadataCatalog.get(os.path.join(args.dataset_name, d)).set(thing_classes=[args.dataset_name])

cfg = get_cfg()
cfg.merge_from_file(args.config)
cfg.DATASETS.TRAIN = (os.path.join(args.dataset_name, "train"),)
cfg.DATASETS.TEST = (os.path.join(args.dataset_name, "val"),)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = args.weight
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 30000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.OUTPUT_DIR = args.output
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
