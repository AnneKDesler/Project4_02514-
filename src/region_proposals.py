import selectivesearch
import cv2
from src.load_data import WASTE, get_dataloaders_WASTE
import numpy as np

def SS(img, size=128):
    """
    Performs Selective Search on an image
    :param img: image to be processed
    :return: img_lbl: image with labels
             regions: regions of the image
    """
    img_lbl, regions = selectivesearch.selective_search(img)#, scale=500, sigma=0.9, min_size=10)
    # crop regions out of the image and resize and append to rps

    return img_lbl, regions


if __name__ == "__main__":
    path = "/u/data/s194333/DLCV/Project4_02514-/data/batch_1/000001.jpg"
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))

    img_lbl, regions = SS(img)
    size = 128

    rps = []
    for r in regions:
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        crop = img[y:y+h, x:x+w]
        crop = cv2.resize(crop, (size, size))
        rps.append(crop)
    
    # save the first 10 regions
    for i, rp in enumerate(rps[:10]):
        cv2.imwrite(f"region_proposals{i}.jpg", rp)

    #for each proposal in each image calc IoU with each GT bbox if above 0.7 assign label, 
    #if more than one label, assign the one with the highest IoU, 
    # if no above 0.3 assign background. discard all others.





