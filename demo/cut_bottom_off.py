import os, argparse
import glob
import cv2

parser = argparse.ArgumentParser(description="Cutting bottom part off an image")
parser.add_argument("--dataset", help="A folder of original images")
parser.add_argument("--output", help="A folder to store new images and labels")
args = parser.parse_args()

cut_point = 9/10

print("The following images contain bbox in the bottom {} part".format(str(1-cut_point)))
for each_img in glob.glob(os.path.join(args.dataset, "*.jpg")):
    each_txt = each_img.replace(".jpg", ".txt")
    img = cv2.imread(each_img)
    lbl = open(each_txt)

    orig = lbl.read()
    lbl.close()
    width, height = img.shape[0], img.shape[1]
    
    img = img[0:int(height*cut_point), 0:width]

    new = []
    for row in orig.split("\n"):
        row = row.split(" ")
        row[2] = str(float(row[2]) / cut_point) # y_center
        if float(row[2]) > 1:
            print(each_img)
            continue
        elif float(row[2]) + float(row[4])/2 > 1:
            row[4] = str((1 - float(row[2])) * 2) # h
        new.append(" ".join(row))

    cv2.imwrite(os.path.join(args.output, os.path.basename(each_img)), img)
    new_lbl = open(os.path.join(args.output, os.path.basename(each_txt)), "w+")
    new_lbl.write("\n".join(new))


