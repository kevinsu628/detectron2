import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
#from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

import random
import os
import numpy as np
import json
import glob
from detectron2.structures import BoxMode
import itertools

# write a function that loads the dataset into detectron2's standard format
def get_dicts(img_dir):
    dataset_dicts = []
    for each_txt in glob.glob(os.path.join(img_dir, "*.txt")):
        record = {}
        
        filename = each_txt.replace(".txt", ".png")
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
      
        #annos = v["regions"]
        objs = []
        txt_reader = open(each_txt, "r")
        for row in txt_reader:
            row = row.strip("\n").split(" ")
            #assert not anno["region_attributes"]
            #anno = anno["shape_attributes"]
            #px = anno["all_points_x"]
            #py = anno["all_points_y"]
            #poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            #poly = list(itertools.chain.from_iterable(poly))
            cls_id = row[0]
            x_c, y_c, w, h = [float(x) for x in row[1:]]
            x_0, y_0 = (x_c - w/2)*width, (y_c - h/2)*height
            x_1, y_1 = (x_c + w/2)*width, (y_c + h/2)*height
            obj = {
                "bbox": [x_0, y_0, x_1, y_1],
                "bbox_mode": BoxMode.XYXY_ABS,
                #"segmentation": [poly],
                "category_id": int(cls_id)
                #"iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
    DatasetCatalog.register("cyc_ped/"+d, lambda d=d: get_dicts("/home/cambricon/Cambricon-MLU100/datasets_old/tsinghua_cyclists/" + d))
    MetadataCatalog.get("cyc_ped/" + d).set(thing_classes=["cyc_ped"])
ts_metadata = MetadataCatalog.get("cyc_ped/train")

#print(ts_metadata)
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("cyc_ped/train",)
cfg.DATASETS.TEST = ("cyc_ped/val",)   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.MODEL.WEIGHTS = "./pretrained_weights/model_final_68b088.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 30000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)
cfg.OUTPUT_DIR = "./output/cyc_ped"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

'''
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_ts_dicts("/home/cambricon/Cambricon-MLU100/datasets_old/Tsinghua_traffic_sign/test_imgs")
print()

for d in random.sample(dataset_dicts, 1): 
    #im = cv2.imread(d["file_name"])
    im = cv2.imread("/home/cambricon/Cambricon-MLU100/datasets_old/COCO/interested_val/000000000724.jpg")
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=ts_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    print(outputs)
    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    records = []
    for each_cls, each_box in zip(pred_classes, pred_boxes):
        cls_id = 0
        [x1, y1, x2, y2] = each_box
        w, h = x2 - x1, y2 - y1
        (img_h, img_w) = outputs["instances"].image_size
        x_center = (x1 + w/2)/img_w
        y_center = (y1 + h/2)/img_h
        w /= img_w
        h /= img_h
        records.append(" ".join([str(x) for x in [cls_id, x_center, y_center, w, h]]))

    jpg_path = "./output_img/"+ (os.path.basename(d["file_name"]))
    cv2.imwrite(jpg_path, im)
    txt_writer = open(jpg_path.replace(".jpg", ".txt"), "w+")
    txt_writer.write("\n".join(records))
    txt_writer.close()
'''
