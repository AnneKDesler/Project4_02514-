import argparse
import pytorch_lightning as pl
import os
from src.load_data import get_dataloaders_proposals
from src.load_data import get_dataloaders_WASTE
from src.model import Model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.load_data import WASTE2
from src.tools import bbox_iou
import numpy as np
import json
from src.tools import nms

def eval(predicts_file):
    # Metrics

    # Same classes
    # Every ground truth should have one proposal. If so, tp +=1.
    # If no proposal matches with iou > 0.7, then fn += 1. But what if other gt with same class matches this one?

    # Different classes
    # If iou > 0.7 but different classes, fp += 1. Or maybe just predicted wrong class?

    # load dictionary from predicts file
    with open(predicts_file) as f:
        data = f.read()
    predicts = json.loads(data)

    tp = 0
    fp = 0
    fn = 0
    tp_binary = 0
    fp_binary = 0
    fn_binary = 0
    mAP = 0
    mAP_binary = 0
    no_of_images = len(predicts.keys())
    for img_id in predicts.keys():
        # for each image

        pred_boxes = predicts[img_id]['pred_bboxes']
        gt_boxes = predicts[img_id]['gt_bboxes']

        pred_boxes = np.array(pred_boxes)
        #gt_boxes = np.array(gt_boxes)
        if pred_boxes.shape[0] == 0:
            continue
        #print(pred_boxes.shape)
        #print(gt_boxes.shape)
        before = pred_boxes.shape[0]
        pred_boxes = nms(pred_boxes, 0.3, 0.5)
        after = pred_boxes.shape[0]
        if before != after:
            print('nms removed ', before-after, ' boxes on image ', img_id)

        pred_boxes=pred_boxes[pred_boxes[:, 4].argsort()]
        len_gts = len(gt_boxes)
        already_found = np.zeros(len_gts)
        already_found_binary = np.zeros(len_gts)
        # for mAP sort pred after confidence score
        AP_xaxis = [[] for _ in range(28)]
        AP_yaxis = [[] for _ in range(28)]
        AP_xaxis_binary = 0
        AP_yaxis_binary = 0
        tp_AP = np.zeros(28)
        tp_AP_binary = 0
        c_pred = np.zeros(28)
        no_in_class = np.zeros(28)
        for gt in gt_boxes:
            no_in_class[int(gt[4])] += 1
        for pred in pred_boxes:
            pred_class = int(pred[5])
            c_pred[pred_class] += 1
            for i in range (len_gts):
                if not already_found[i]:
                    gt = gt_boxes[i]
                    iou = bbox_iou(gt[:4],pred[:4])
                    if iou > 0.5:
                        if not already_found_binary[i]:
                            tp_binary += 1
                            tp_AP_binary += 1
                            already_found_binary[i] = 1
                        if pred_class == gt[4]:
                            tp += 1
                            tp_AP[pred_class] += 1
                            already_found[i] = 1
            if already_found[i] == 0:
                fp += 1
            if already_found_binary[i] == 0:
                fp_binary += 1
            if no_in_class[pred_class] != 0:
                AP_xaxis[pred_class].append(tp_AP[pred_class]/c_pred[pred_class])
                AP_yaxis[pred_class].append(tp_AP[pred_class]/(no_in_class[pred_class]))
            AP_yaxis_binary += tp_AP_binary/len_gts
            if tp_AP_binary/len_gts > 1:
                print('tp_binary:' , tp_AP_binary, 'len_gts:', len_gts)

        fn += len_gts - already_found.sum()
        fn_binary += len_gts - already_found_binary.sum()
        AP_image = 0
        active_classes = 0
        for i in range(28):
            if len(AP_xaxis[i])>0:
                active_classes += 1
                # Remove non-unique Recall observations
                AP_xaxis[i],indices = np.unique(AP_xaxis[i],return_index=True)
                tmp = []
                for j in indices:
                    tmp.append(AP_yaxis[i][j])
                AP_yaxis[i] = tmp
                AP_image += np.mean(AP_yaxis[i])
        
        if active_classes > 0:
            AP_image /= active_classes
        mAP += AP_image
        mAP_binary += AP_yaxis_binary

    # after we went through images
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    precision_binary = tp_binary/(tp_binary+fp_binary)
    recall_binary = tp_binary/(tp_binary+fn_binary)
    dice = 2*tp/(2*tp+fp+fn)
    dice_binary = 2*tp_binary/(2*tp_binary+fp_binary+fn_binary)
    print('mAP: ', mAP, 'mAP_binary: ', mAP_binary, 'no_of_images: ', no_of_images)

    mAP /= no_of_images
    mAP_binary /= no_of_images
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Precision binary: ", precision_binary)
    print("Recall binary: ", recall_binary)
    print("Dice score: ", dice)
    print("Dice score binary: ", dice_binary)
    print("mAP: ", mAP)
    print("mAP binary: ", mAP_binary)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="outputs_nms/predicts_all.txt",
        type=str,
        help="path to ckpt file to evaluate",
    )

    args = parser.parse_args()

    eval(args.path)
