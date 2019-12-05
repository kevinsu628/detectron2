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
import os, shutil
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
    parser.add_argument(
        "--copy_old_labels",
        help="if 1, will copy the old label files in dataset folder to output folder, and append\
 newly generated labels to the old files. O.w., will create a new label. ",
        type=int
    )
    parser.add_argument(
        "--copy_imgs",
        help="if 1, will copy the image into output folder",
        type=int
    )
    parser.add_argument(
        "--mode",
        help="The weight predicting coco cls or cyclist",
        type=str
    )
    return parser


def parsePrediction(outputs, mode):
    # person bike car motor bus truck traffic light
    #interested_cls = [0,1,2,3,5,7,9]
    #interested_cls_map = [0,1,2,3,4,5,6]
    interested_cls = [1,3,5,7,9]
    interested_cls_map = [1,3,4,5,6]
    cyclist_cls = 8
    ts_cls = 7 
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    records = []
    for each_cls, each_box in zip(pred_classes, pred_boxes):    
        cls_id = each_cls

        if mode == "coco": 
            if int(each_cls) in interested_cls:
                cls_id = interested_cls_map[interested_cls.index(int(each_cls))]
            else:
                continue
        elif mode == "cyclist":
            cls_id = cyclist_cls
        elif mode == "ts": 
            cls_id = ts_cls    
   
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
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1#80
cfg.OUTPUT_DIR = args.output
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set the testing threshold for this model
predictor = DefaultPredictor(cfg)


if predict_whole_folder:
    for d in glob.glob(os.path.join(args.dataset, "*"+args.ext)):
        im = cv2.imread(d)
        txt_name = os.path.basename(d).replace("."+args.ext, ".txt")
        start_time = time.time()
        outputs = predictor(im)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                d, len(outputs["instances"]), time.time() - start_time
            )
        )

        records = parsePrediction(outputs, mode=args.mode)
        if args.copy_imgs:
            jpg_path = os.path.join(args.output, os.path.basename(d))
            cv2.imwrite(jpg_path, im)
        txt_path = os.path.join(args.dataset, txt_name)
        new_txt_path = os.path.join(args.output, txt_name)
        if args.copy_old_labels and os.path.exists(txt_path):
            old_txt_reader = open(txt_path, "r")
            old_records = [r.strip("\n") for r in old_txt_reader]
            old_txt_reader.close()
            txt_writer = open(new_txt_path, "w+")
            txt_writer.write("\n".join(old_records + records))
            txt_writer.close()
        else:
            txt_writer = open(new_txt_path, "w+")
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
