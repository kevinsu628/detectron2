import os
import glob
import cv2

img = cv2.imread("/home/cambricon/Cambricon-MLU100/datasets_old/Tsinghua_traffic_sign/auto_label/ts_coco_cyc_adjusted_label/52020.jpg")
lbl = open("/home/cambricon/Cambricon-MLU100/datasets_old/Tsinghua_traffic_sign/auto_label/ts_coco_cyc_adjusted_label/52020.txt", "r")

orig = lbl.read()
lbl.close()
width, height = img.shape[0], img.shape[1]
cut_point = 9/10
img = img[0:int(height*cut_point), 0:width]

new = []
for row in orig.split("\n"):
    row = row.split(" ")
    x_c, y_c, w, h = row[1:]
    #y1_rel = float(y_c) - float(h)/2
    #y2_rel = float(y_c) + float(h)/2
    #if y1_rel > cut_point: #y1
        # drop this bbox
    #    print("Cut off") 
    #    continue
    #elif y2_rel > cut_point: #y2
        # modify y2 to be cut_point
    #    row[4] = str((float(row[4]) - (1 - cut_point))/cut_point)
    row[2] = str(float(row[2]) / cut_point)
    if float(row[2]) > 1:
        continue
    elif float(row[2]) + float(row[4])/2 > 1:
        row[4] = str((1 - float(row[2])) * 2)
    new.append(" ".join(row))
print("\n".join(new))
    

cv2.imwrite("./temp/ttest.jpg", img)
new_lbl = open("./temp/ttest.txt", "w+")
new_lbl.write("\n".join(new))

