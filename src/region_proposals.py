import selectivesearch
import cv2
from src.load_data import WASTE, get_dataloaders_WASTE
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import data_augment as dataAug

def SS(img, size=128):
    """
    Performs Selective Search on an image
    :param img: image to be processed
    :return: img_lbl: image with labels
             regions: regions of the image
    """
    img_lbl, regions = selectivesearch.selective_search(img)#, scale=500, sigma=0.9, min_size=10)
    # crop regions out of the image and resize and append to rps
    # regions [x, y, w, h] upper left corner and width and height
    # changed to center x, center y, width, height
    for i, r in enumerate(regions):
        x, y, w, h = r['rect']
        regions[i]['rect'] = [x + w//2, y + h//2, w, h]
    

    return regions


if __name__ == "__main__":
    path = "/u/data/s194333/DLCV/Project4_02514-/data"
    train_loader, val_loader, test_loader = get_dataloaders_WASTE(batch_size=1, num_workers=0, seed=42, data_path=path)
    img, target, img_id = next(iter(test_loader))
    img = img.squeeze(0).permute(1,2,0).numpy()
    # save image
    fig,ax = plt.subplots(1)
    plt.axis('off')
    ax.imshow(img)
    fig.savefig(str(img_id.item())+'.png', bbox_inches='tight', pad_inches=0)
        

    print(img.shape)

    regions = SS(img)
    size = 128

    rps = []
    for r in regions:
        x, y, w, h = r['rect']
        if w <= 1 or h <= 1:
            continue
        #print(x, y, w, h)
        crop = img[y-h//2:y+h//2, x-w//2:x+w//2,:].astype(np.float32)
        crop = dataAug.Resize((size, size),False)(np.copy(crop))
        rps.append(crop)
    
    # save the first 10 regions
    for i, rp in enumerate(rps[:10]):

        cv2.imwrite(f"region_proposals{i}.jpg", rp)
        fig,ax = plt.subplots(1)
        plt.axis('off')
        ax.imshow(rp)
        fig.savefig(f"region_proposals{i}.jpg", bbox_inches='tight', pad_inches=0)
            


    #for each proposal in each image calc IoU with each GT bbox if above 0.7 assign label, 
    #if more than one label, assign the one with the highest IoU, 
    # if no above 0.3 assign background. discard all others.





