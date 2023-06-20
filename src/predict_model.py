import argparse
import pytorch_lightning as pl
import torch
import os
from src.model import Model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.load_data import WASTE2
from src.load_data import get_dataloaders_proposals

def predict(model_src):
    if not os.path.isfile(model_src):
        model_src = os.path.join("models/lr0.0001", model_src)

    model = Model.load_from_checkpoint(checkpoint_path=model_src)

    path = "/u/data/s194333/DLCV/Project4_02514-/data"
    dataset = WASTE2(data_path=path)
    cats = dataset.super_cat_ids

    trainloader, valloader, testloader = get_dataloaders_proposals(batch_size=1, data_path="data", proposal_path="region_proposals5")

    model.to("cuda")

    prev_img_id = -1
    for prop, rect, img_id in testloader:
        if img_id.item() != prev_img_id:
            img, target, _= dataset.__getitem__(img_id.item())
            fig,ax = plt.subplots(1)
            plt.axis('off')
            ax.imshow(img)#.squeeze(0))#.permute(1,2,0))
            for box in target:
                gt = Rectangle((box[0]-box[2]/2,box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(gt)
                # get keyword argument from super category id
                label = list(cats.keys())[list(cats.values()).index(box[4].item())]
                plt.text(box[0]+box[2]/2,box[1]-box[3]/2,str(label),color='r')
        prev_img_id = img_id.item()

        output = model(prop.to("cuda"))
        #output = torch.softmax(output, dim=1)

        output = torch.sigmoid(output)

        pred = torch.argmax(output.squeeze(0), dim=0)
        pred2 = torch.argmax(output.squeeze(0)[:-1], dim=0)
        print(output[0,pred], output[0,pred2])
        if pred != 28 or output[0,pred2] >= 0.3:
            
            box = rect[0] 
            gt = Rectangle((box[0]-box[2]/2,box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(gt)
            label = list(cats.keys())[list(cats.values()).index(pred2)] 
            label = label + ' ' + str(round(output.squeeze(0)[pred2].item(),2))
            plt.text(box[0]+box[2]/2,box[1]-box[3]/2,str(label),color='b')
            
            fig.savefig('outputs/' +str(img_id.item())+'_output.png', bbox_inches='tight', pad_inches=0)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="best-v1.ckpt",
        type=str,
        help="path to ckpt file to evaluate",
    )

    args = parser.parse_args()

    predict(args.path)
