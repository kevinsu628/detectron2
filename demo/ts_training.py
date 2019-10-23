import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
import numpy as np
import json
import glob
from detectron2.structures import BoxMode
import itertools

# write a function that loads the dataset into detectron2's standard format
def get_ts_dicts(img_dir):
    dataset_dicts = []
    for each_txt in glob.glob(os.path.join(img_dir, "*.txt")):
        record = {}
        
        filename = each_txt.replace(".txt", ".jpg")
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
            if row[0] != "7": 
                continue
            x_c, y_c, w, h = row[1:]
            x_0, y_0 = x_c - w/2, y_c - h/2
            obj = {
                "bbox": [x_0, y_0, w, h],
                "bbox_mode": BoxMode.XYWH_REL,
                #"segmentation": [poly],
                "category_id": 0
                #"iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
    DatasetCatalog.register("ts/" + d, lambda d=d: get_ts_dicts("ts/" + d))
    MetadataCatalog.get("ts/" + d).set(thing_classes=["ts"])
ts_metadata = MetadataCatalog.get("ts/train")


import random

dataset_dicts = get_ts_dicts("/home/kevin/ascent/dataset/Tsinghua_traffic_sign/all_images/")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=ts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])