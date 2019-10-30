# Parameter:
# config-file: path to cfg file
# weight_path: path to the pretrained weight
# dataset_path: path to a directory of images

# This script predicts bboxes of every image in the dataset path,
# write the ground truth into yolo format .txt filess

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 predict whole folder")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--dataset_folder", help="A folder of images to predict")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    while True:
    #for each_img in glob.glob(os.path.join(args.dataset_folder, "*.jpg")):
        # use PIL, to be consistent with evaluation
        each_img = "/home/cambricon/Cambricon-MLU100/datasets_old/Tsinghua_traffic_sign/test_imgs/41372.jpg"
        each_img = "/home/cambricon/Cambricon-MLU100/datasets_old/COCO/interested_val/000000011197.jpg"
        img = read_image(each_img, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                each_img, len(predictions["instances"]), time.time() - start_time
            )
        )

        pred_classes = predictions["instances"].pred_classes.cpu().numpy()
        pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        records = []
        print(pred_boxes)
        for each_cls, each_box in zip(pred_classes, pred_boxes):
            if int(each_cls) in interested_cls:
                cls_id = interested_cls.index(int(each_cls)) 
                [x1, y1, x2, y2] = each_box
                w, h = x2 - x1, y2 - y1
                (img_h, img_w) = predictions["instances"].image_size
                x_center = (x1 + w/2)/img_w
                y_center = (y1 + h/2)/img_h
                w /= img_w
                h /= img_h
                records.append(" ".join([str(x) for x in [cls_id, x_center, y_center, w, h]]))
        break
        #each_txt = each_img.replace(".jpg", ".txt")
        #txt_writer = open(each_txt, "a+")
        #txt_writer.write("\n".join(records) + "\n") 
             
                

