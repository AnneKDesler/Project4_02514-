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

        len_gts = len(gt_boxes)
        already_found = np.zeros(len_gts)
        already_found_binary = np.zeros(len_gts)
        # for mAP sort pred after confidence score
        mAP_xaxis = [[] for _ in range(28)]
        mAP_yaxis = [[] for _ in range(28)]
        tp_mAP = np.zeros(len_gts)
        c_pred = np.zeros(len_gts)
        no_in_class = np.zeros(len_gts)
        for gt in gt_boxes:
            no_in_class[gt[5]] += 1
        for pred in pred_boxes:
            pred_class = pred[5]
            c_pred[pred_class] += 1
            for i in range (len_gts):
                if not already_found[i]:
                    gt = gt_boxes[i]
                    iou = bbox_iou(gt[:4],pred[:4])
                    if iou > 0.5:
                        if not already_found_binary[i]:
                            tp_binary += 1
                            already_found_binary[i] = 1
                        if pred_class == gt[5]:
                            tp += 1
                            tp_mAP[pred_class] += 1
                            already_found[i] = 1
            if already_found[i] == 0:
                fp += 1
            if already_found_binary[i] == 0:
                fp_binary += 1
            if no_in_class[pred_class] != 0:
                mAP_xaxis[pred_class].append(tp_mAP[pred_class]/c_pred[pred_class])
                mAP_yaxis[pred_class].append(tp[pred_class]/no_in_class[pred_class])

        fn += len_gts - already_found.sum()
        fn_binary += len_gts - already_found_binary.sum()

        mAP_image = 0
        active_classes = 0
        for i in range(28):
            if no_in_class != 0:
                active_classes += 1
                # Remove non-unique Recall observations
                mAP_xaxis[i],indices = np.unique(mAP_xaxis[i],return_index=True)
                tmp = []
                for j in indices:
                    tmp.append(mAP_yaxis[i][j])
                mAP_yaxis[i] = tmp

                mAP_image += np.mean(mAP_yaxis[i])
        mAP_image /= active_classes
        mAP += mAP_image

    # after we went through images
    dice = 2*tp/(2*tp+fp+fn)
    dice_binary = 2*tp_binary/(2*tp_binary+fp_binary+fn_binary)
    mAP /= no_of_images
    print("Dice score: ", dice)
    print("Dice score binary: ", dice_binary)
    print("mAP: ", mAP)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="outputs_nms/predicts.txt",
        type=str,
        help="path to ckpt file to evaluate",
    )

    args = parser.parse_args()

    eval(args.path)
