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
from detectron2.utils.visualizer import ColorMode

import time
import random
import os
import glob
from detectron2.structures import BoxMode


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 predicting whole folder or an image")
    parser.add_argument("--dataset", help="A folder of images to predict, or a path to an image")
    parser.add_argument("--output", help="A folder to store predicted txts in yolo format")
    parser.add_argument(
        "--config",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        help="path to config file"
    )
    parser.add_argument(
        "--weight",
        help="Pretrained weight"
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

def parsePrediction(outputs):
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    records = []
    for each_cls, each_box in zip(pred_classes, pred_boxes):
        cls_id = each_cls
        [x1, y1, x2, y2] = each_box
        w, h = x2 - x1, y2 - y1
        (img_h, img_w) = outputs["instances"].image_size
        x_center = (x1 + w / 2) / img_w
        y_center = (y1 + h / 2) / img_h
        w /= img_w
        h /= img_h
        records.append(" ".join([str(x) for x in [cls_id, x_center, y_center, w, h]]))
    return records


args = get_parser().parse_args()
logger = setup_logger()
predict_whole_folder = os.path.isdir(args.dataset)

cfg = get_cfg()
cfg.merge_from_file(args.config)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = args.weight
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 30000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = args.output
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)


if predict_whole_folder:
    for d in glob.glob(os.path.join(args.dataset, "*"+args.ext)):
        im = cv2.imread(d)
        #start_time = time.time()
        outputs = predictor(im)
        #logger.info(
        #    "{}: detected {} instances in {:.2f}s".format(
        #        d, len(outputs["instances"]), time.time() - start_time
        #    )
        #)
        records = parsePrediction(outputs)
        jpg_path = os.path.join(args.output, os.path.basename(d))
        cv2.imwrite(jpg_path, im)
        txt_writer = open(jpg_path.replace("."+args.ext, ".txt"), "w+")
        txt_writer.write("\n".join(records))
        txt_writer.close()
else:
    im = cv2.imread(args.dataset)
    outputs = predictor(im)
    print(outputs)
    records = parsePrediction(outputs)
    jpg_path = os.path.join(args.output, os.path.basename(args.dataset))
    cv2.imwrite(jpg_path, im)
    txt_writer = open(jpg_path.replace("." + args.ext, ".txt"), "w+")
    txt_writer.write("\n".join(records))
    txt_writer.close()
