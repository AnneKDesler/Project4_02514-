import selectivesearch
import cv2
from src.load_data import WASTE, get_dataloaders_WASTE
import numpy as np
from torch.utils.data import DataLoader
from src.region_proposals import SS
import matplotlib.pyplot as plt
import os
from tools import iou_xywh_numpy
import torch

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = "/u/data/s194333/DLCV/Project4_02514-/data"
    list = ["train", "val", "test"]
    for mode in list:
            
        dataset = WASTE(data_path=path, mode = mode)
        reg_path = "/u/data/s194333/DLCV/Project4_02514-/region_proposals5"+"/"+mode
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        k = 0
        if not os.path.exists(reg_path):
            os.makedirs(reg_path)
        for img, target, img_id in data_loader:

            regions = SS(img.squeeze(0).permute(1,2,0).numpy())
            # save all regions

            rects = []
            back_rects = []
            for i, r in enumerate(regions):
                rect = np.array(r['rect'])
                #for each proposal in each image calc IoU with each GT bbox if above 0.7 assign label,
                #if more than one label, assign the one with the highest IoU,
                # if no above 0.3 assign background. discard all others.

                ious = np.zeros(28)
                for j, bbox in enumerate(target[0]):
                    gt = bbox.numpy()
                    iou = iou_xywh_numpy(rect, gt[:4])
                    ious[int(gt[4])] = iou
                
                argmax_iou = np.argmax(ious)
                max_iou = ious[argmax_iou]

                if max_iou > 0.7:
                    rects.append(np.append(np.append(rect, argmax_iou), img_id.item()))     
                elif max_iou < 0.3:

                    back_rects.append(np.append(np.append(rect, 28), img_id.item()))
            
            for rect in rects:
                np.save(reg_path + "/" + str(k) +".npy", rect)
                k += 1
            
            num_back_ground = 3*len(rects)
            if len(back_rects) > num_back_ground:
                idxs = np.random.choice(len(back_rects), num_back_ground, replace=False)
                for idx in idxs:
                    np.save(reg_path + "/" + str(k) +".npy", back_rects[idx])
                    k += 1

        
        
        
