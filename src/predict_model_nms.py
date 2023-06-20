import argparse
import pytorch_lightning as pl
import torch
import os
from src.model import Model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.load_data import WASTE2
from src.load_data import get_dataloaders_proposals
import numpy as np
from tools import nms

def predict(model_src):
    if not os.path.isfile(model_src):
        model_src = os.path.join("models/lr0.0001", model_src)

    model = Model.load_from_checkpoint(checkpoint_path=model_src)

    path = "/u/data/s194333/DLCV/Project4_02514-/data"
    dataset = WASTE2(data_path=path)
    cats = dataset.super_cat_ids

    trainloader, valloader, testloader = get_dataloaders_proposals(batch_size=1, data_path="data", proposal_path="region_proposals5")

    model.to("cuda")
    
    predicts = dict()
    prev_img_id = -1
    i = 0
    for prop, rect, img_id in testloader:
        i += 1
        if img_id.item() != prev_img_id:
            img, target, _= dataset.__getitem__(img_id.item())
            predicts[str(img_id.item())] = dict()
            predicts[str(img_id.item())]['gt_bboxes'] = []
            predicts[str(img_id.item())]['pred_bboxes'] = []
            fig,ax = plt.subplots(1)
            plt.axis('off')
            ax.imshow(img)#.squeeze(0))#.permute(1,2,0))
            for box in target:
                box = box.tolist()
                predicts[str(img_id.item())]['gt_bboxes'].append(box)
            
            
        prev_img_id = img_id.item()

        output = model(prop.to("cuda"))
        #output = torch.softmax(output, dim=1)

        output = torch.sigmoid(output)
        output = output.cpu().detach()

        pred = torch.argmax(output.squeeze(0), dim=0)
        pred2 = torch.argmax(output.squeeze(0)[:-1], dim=0)
        print(output[0,pred], output[0,pred2])
        if pred != 28 or output[0,pred2] >= 0.3:
            box = rect[0][:4].tolist()

            box.append(output[0,pred2])
            box.append(pred2)
            predicts[str(img_id.item())]['pred_bboxes'].append(box)

        
 
    for img_id in predicts.keys():
        fig,ax = plt.subplots(1)
        plt.axis('off')
        img, _, _ = dataset.__getitem__(int(img_id))
        ax.imshow(img)
        for box in predicts[img_id]['gt_bboxes']:
            gt = Rectangle((box[0]-box[2]/2,box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(gt)
            # get keyword argument from super category id
            label = list(cats.keys())[list(cats.values()).index(box[4])]
            plt.text(box[0]+box[2]/2,box[1]-box[3]/2,str(label),color='r')
        
        pred_boxes = predicts[img_id]['pred_bboxes']
        pred_boxes = np.array(pred_boxes)
        if pred_boxes.shape[0] == 0:
            continue
        before = pred_boxes.shape[0]
        pred_boxes = nms(pred_boxes, 0.3, 0.5)
        after = pred_boxes.shape[0]
        if before != after:
            print('nms removed ', before-after, ' boxes on image ', img_id)
        for box in pred_boxes:
            gt = Rectangle((box[0]-box[2]/2,box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(gt)
            # get keyword argument from super category id
            label = list(cats.keys())[list(cats.values()).index(box[5])]
            label = label + ' ' + str(round(box[4],2))
            plt.text(box[0]+box[2]/2,box[1]-box[3]/2,str(label),color='b')
        fig.savefig('outputs_nms/' +str(img_id)+'_output_nms.png', bbox_inches='tight', pad_inches=0)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="best.ckpt",
        type=str,
        help="path to ckpt file to evaluate",
    )

    args = parser.parse_args()

    predict(args.path)
