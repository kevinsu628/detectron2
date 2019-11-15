import os, glob, argparse, shutil

parser = argparse.ArgumentParser(description="Solve bbox conflict in a dataset auto labelled by models.")
parser.add_argument("--dataset", help="A folder of images and its yolo labels")
parser.add_argument("--output", help="The output folder to store copy of images and its adjusted yolo labels")
args = parser.parse_args()


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def bboxRelToX1Y1X2Y2(row):
    [x_c, y_c, w, h] = row
    x_0, y_0 = (x_c - w / 2), (y_c - h / 2)
    x_1, y_1 = (x_c + w / 2), (y_c + h / 2)
    return (x_0, y_0, x_1, y_1)

# key: cls_id
# item: lst of bboxes
def parseYoloTxt(each_txt):
    txt_reader = open(each_txt, "r")
    records = {}
    for i in range(9): # cls_id: 0~8
        records[str(i)] = []
    for row in txt_reader:
        r = row.strip("\n").split(" ") 
        records[r[0]].append(tuple([float(x) for x in r[1:]]))
    return records

# return a list of bboxes that has IOU higher than thres with the bbox of obj
def getBboxWithIOUGreaterThan(obj, bboxes, thres): 
    ret = []
    for each_bbox in bboxes:
        val = iou(bboxRelToX1Y1X2Y2(obj), bboxRelToX1Y1X2Y2(each_bbox))
        if val > thres:
            ret.append(each_bbox)
    return ret


# solve bbox overlapping conflicts between classes (person, bike, cyclists)
def solveCyclistConflict(cyclists, bikes, person, vehicles):
    removed_cyclists = []
    removed_bikes = []
    removed_persons = []

    for each_cyclist_bbox in cyclists:
        overlaped_persons = getBboxWithIOUGreaterThan(each_cyclist_bbox, person, 0.2)
        overlaped_bikes = getBboxWithIOUGreaterThan(each_cyclist_bbox, bikes, 0.1)
        overlaped_vehicles = getBboxWithIOUGreaterThan(each_cyclist_bbox, vehicles, 0.8)

        removed_persons += overlaped_persons
        removed_bikes += overlaped_bikes
        # Cyclists auto labeller sometimes labels vehicles as a cyc.
        # This is caused by tricycle class from tsinghua cyclists dataset
        if len(overlaped_vehicles) > 0:
            removed_cyclists.append(each_cyclist_bbox)

    kept_cyclists =  list(set(cyclists) - set(removed_cyclists))
    kept_person =  list(set(person) - set(removed_persons))
    kept_bikes =  list(set(bikes) - set(removed_bikes))
    
    return kept_cyclists, kept_person, kept_bikes

# solve bbox overlapping conflicts between classA and classB
# this function keeps the bbox of classA and drop classB if A and B overlaps > thres
def solveOverlappingConflict(classA, classB, thres):
    removed_classB = []
    for bboxA in classA:
        overlaped = getBboxWithIOUGreaterThan(bboxA, classB, 0.9)
        removed_classB.append(overlaped)
    kept_classB = list(set(classB) - set(removed_classB))
    return kept_classB

# a record is a list of tuple with yolo format bbox
# (x_c, y_c, w, h)
def recordsToRow(records, cls_id):
    ret = []
    for each_bb in records:
        ret.append(" ".join([str(x) for x in [cls_id] + list(each_bb)]))
    return ret

# this script assumes the dataset has labels from 3 labellers:
# traffic sign, COCO, cyclists
# thus, contains 9 classes 
# person, bicycle, car, motorcycle, bus, truck, traffic light, traffic sign, cyclists
if __name__ == "__main__":
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for each_txt in glob.glob(os.path.join(args.dataset, "*.txt")): 
        records = parseYoloTxt(each_txt)

        cyclists = records["8"]
        bikes = records["1"] + records["3"]
        person = records["0"]
        vehicles = records["2"] + records["4"] + records["5"] # use all vehicles to solve cyclist conflict
        kept_cyclists, kept_person, kept_bikes = solveCyclistConflict(cyclists, bikes, person, vehicles)
        kept_cars_ = solveOverlappingConflict(records["4"], records["2"], 0.9)
        kept_cars = solveOverlappingConflict(records["5"], kept_cars_, 0.9)

        ###### Combine into new records ######
        # since we combined bike and motor. They both have cls_id 1. 
        # cyclist will have cls_id 3
        new_records = []

        # cyclists, pedestrian, bikes, cars 
        modified_ids = [3, 0, 1, 2]
        for bb, cls_id in zip([kept_cyclists, kept_person, kept_bikes, kept_cars], modified_ids):
            new_records += recordsToRow(bb, cls_id)

        for k in records.keys():
            if int(k) not in modified_ids + [8]:
                new_records += recordsToRow(records[k], k)

        jpg_path = each_txt.replace(".txt", ".jpg")
        txt_writer = open(os.path.join(args.output, os.path.basename(each_txt)), "w+")
        txt_writer.write("\n".join(new_records))
        txt_writer.close()
        shutil.copyfile(jpg_path, (os.path.join(args.output, os.path.basename(jpg_path))))
        
