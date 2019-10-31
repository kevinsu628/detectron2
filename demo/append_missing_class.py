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
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)
    ### END CODE HERE ###

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###

    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area / union_area
    ### END CODE HERE ###

    return iou

# row: lst. [cls, x_c, y_c, w, h]
def bboxRelToAbs(row, img_w, img_h):
    x_c, y_c, w, h = [float(x) for x in row[1:]]
    x_0, y_0 = (x_c - w / 2) * img_w, (y_c - h / 2) * img_h
    x_1, y_1 = (x_c + w / 2) * img_w, (y_c + h / 2) * img_h
    return x_0, y_0, x_1, y_1

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
            x_0, y_0, x_1, y_1 = bboxRelToAbs(row, width, height)
            obj = {
                "bbox": [x_0, y_0, x_1, y_1],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(cls_id)
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def parsePrediction(outputs, orig_rows):
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    records = []
    for each_cls, each_box in zip(pred_classes, pred_boxes):
        cls_id = each_cls

        if str(cls_id) != "1":
            continue

        [x1, y1, x2, y2] = each_box # prediction 
        w, h = x2 - x1, y2 - y1
        (img_h, img_w) = outputs["instances"].image_size

        # check if the predicted bbox overlap with any existing bbox by checking IOU > 0.7
        no_overlap = True
        for row in orig_rows:
            row = row.split(" ")
            x1_, y1_, x2_, y2_ = bboxRelToAbs(row, img_w, img_h)
            if iou((x1,y1,x2,y2), x1_,y1_,x2_,y2_) > 0.5:
                no_overlap = False
                break
        if no_overlap:
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
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.OUTPUT_DIR = args.output
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)


if predict_whole_folder:
    for d in glob.glob(os.path.join(args.dataset, "*"+args.ext)):
        im = cv2.imread(d)
        start_time = time.time()
        outputs = predictor(im)

        each_txt = d.replace("*"+args.ext, ".txt")
        txt_reader = open(each_txt, "r")
        orig_rows = [r.strip("\n") for r in txt_reader]

        records = parsePrediction(outputs, orig_rows) # additional predictions 
        jpg_path = os.path.join(args.output, os.path.basename(d))
        cv2.imwrite(jpg_path, im)
        txt_writer = open(jpg_path.replace("."+args.ext, ".txt"), "a+")
        txt_writer.write("\n".join(records))
        txt_writer.close()
else:
    im = cv2.imread(args.dataset)
    outputs = predictor(im)

    each_txt = d.replace("*"+args.ext, ".txt")
    txt_reader = open(each_txt, "r")
    orig_rows = [r.strip("\n") for r in txt_reader]

    records = parsePrediction(outputs, orig_rows) # additional predictions 
    print(records)
    jpg_path = os.path.join(args.output, os.path.basename(d))
    cv2.imwrite(jpg_path, im)
    txt_writer = open(jpg_path.replace("."+args.ext, ".txt"), "a+")
    txt_writer.write("\n".join(records))
    txt_writer.close()
