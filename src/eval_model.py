import argparse
import pytorch_lightning as pl
import torch
import os
from src.load_data import get_dataloaders_proposals
from src.load_data import get_dataloaders_WASTE
from src.model import Model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.load_data import WASTE2
from src.tools import bbox_iou


def eval(model_src):
    if not os.path.isfile(model_src):
        model_src = os.path.join("models/lr0.0001", model_src)

    model = Model.load_from_checkpoint(checkpoint_path=model_src)

    path = "/u/data/s194333/DLCV/Project4_02514-/data"
    dataset = WASTE2(data_path=path)
    cats = dataset.super_cat_ids

    trainloader, valloader, testloader = get_dataloaders_proposals(batch_size=1, data_path="data", proposal_path="region_proposals5")

    model.to("cuda")

    tp = 0
    fp = 0
    fn = 0

    for prop, rect, img_id in testloader:
        img, target, _= dataset.__getitem__(img_id.item())

        output = model(prop.to("cuda"))
        #output = torch.softmax(output, dim=1)

        output = torch.sigmoid(output)

        pred = torch.argmax(output.squeeze(0), dim=0)
        pred2 = torch.argmax(output.squeeze(0)[:-1], dim=0)
        print(output[0,pred], output[0,pred2])
        if pred != 28 or output[0,pred2] >= 0.3:
            fig,ax = plt.subplots(1)
            plt.axis('off')
            ax.imshow(img)#.squeeze(0))#.permute(1,2,0))
            for box in target:
                gt = Rectangle((box[0]-box[2]/2,box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(gt)
                # get keyword argument from super category id
                label = list(cats.keys())[list(cats.values()).index(box[4].item())]
                plt.text(box[0]+box[2]/2,box[1]-box[3]/2,str(label),color='r')
            
            box = rect[0] 
            preddddd = Rectangle((box[0]-box[2]/2,box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(gt)
            # get keyword argument from super category id
            #if pred != 28:
            #    label = list(cats.keys())[list(cats.values()).index(pred2)] 
            #else:
            #    label = 'background'
            label = list(cats.keys())[list(cats.values()).index(pred2)] 
            label = label + ' ' + str(round(output.squeeze(0)[pred2].item(),2))
            plt.text(box[0]+box[2]/2,box[1]-box[3]/2,str(label),color='b')
            
            fig.savefig('outputs/' +str(img_id.item())+'_output.png', bbox_inches='tight', pad_inches=0)

            fig,ax = plt.subplots(1)
            plt.axis('off')
            ax.imshow(prop.squeeze(0).permute(1,2,0))
            fig.savefig('outputs/' +str(img_id.item())+'_prop.png', bbox_inches='tight', pad_inches=0)


# Metrics

# Same classes
# Every ground truth should have one proposal. If so, tp +=1.
# If no proposal matches with iou > 0.7, then fn += 1. But what if other gt with same class matches this one?

# Different classes
# If iou > 0.7 but different classes, fp += 1. Or maybe just predicted wrong class?
import numpy as np

tp = 0
fp = 0
fn = 0
tp_binary = 0
fp_binary = 0
fn_binary = 0
mAP = 0
mAP_binary = 0
# for each image
len_gts = len(gts)
already_found = np.zeros(len_gts)
already_found_binary = np.zeros(len_gts)
# for mAP sort pred after confidence score
mAP_xaxis = [[] for _ in range(28)]
mAP_yaxis = [[] for _ in range(28)]
tp_mAP = np.zeros(len_gts)
c_pred = np.zeros(len_gts)
no_in_class = np.zeros(len_gts)

pred_boxes = predicts[img_id]['pred_bboxes']
gt_boxes = predicts[img_id]['gt_bboxes']
for gt in gt_boxes:
    no_in_class[gt[5]] += 1
for pred in pred_boxes:
    pred_class = pred[5]
    c_pred[pred_class] += 1
    for i in range (len_gts):
        if not already_found[i]:
            gt = gts[i]
            iou = bbox_iou(gt[:3],pred[:3])
            if iou > 0.5:
                if not already_found_binary[i]:
                    tp_binary += 1
                    already_found_binary[i] = 1
                if pred_class == gt[5]:
                    tp += 1
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="best.ckpt",
        type=str,
        help="path to ckpt file to evaluate",
    )

    args = parser.parse_args()

    eval(args.path)
