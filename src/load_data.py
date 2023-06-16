import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import os
import glob
import numpy as np
import PIL.Image as Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json
import pandas as pd
import seaborn as sns; sns.set()
from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.collections import PatchCollection
import random
import pylab
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import data_augment as dataAug

class WASTE(torch.utils.data.Dataset):
    def __init__(self, data_path='/dtu/datasets1/02514/data_wastedetection', img_size=512):
        'Initialization'
        self.img_size = img_size
        self.path = data_path
        self.ann_path = data_path + '/annotations.json'

        with open(self.ann_path, 'r') as f:
            dataset = json.loads(f.read())

        categories = dataset['categories']
        self.anns = dataset['annotations']
        self.imgs = dataset['images']
        self.img_ids = []
        for img in self.imgs:
            self.img_ids.append(img['id'])

        cat_names = []
        super_cat_names = []
        super_cat_ids = {}
        super_cat_last_name = ''
        nr_super_cats = 0
        for cat_it in categories:
            cat_names.append(cat_it['name'])
            super_cat_name = cat_it['supercategory']
            # Adding new supercat
            if super_cat_name != super_cat_last_name:
                super_cat_names.append(super_cat_name)
                super_cat_ids[super_cat_name] = nr_super_cats
                super_cat_last_name = super_cat_name
                nr_super_cats += 1

        self.super_cat_ids = super_cat_ids



        self.cat_ids_2_supercat_ids = {}
        for cat in categories:
            self.cat_ids_2_supercat_ids[cat['id']] = super_cat_ids[cat['supercategory']]

        
    
    def __len__(self):
        'Returns the total number of samples'
        return len(self.img_ids)
    

    def __getitem__(self, idx):
        'Generates one sample of data'

        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        # Loads dataset as a coco object
        coco = COCO(self.ann_path)

        # Find image id
        img_id = self.img_ids[idx]

        img_path = self.imgs[img_id]['file_name']

        I = Image.open(self.path + '/' + img_path)
        # Load and process image metadata
        if I._getexif():
            exif = dict(I._getexif().items())
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    I = I.rotate(180,expand=True)
                if exif[orientation] == 6:
                    I = I.rotate(270,expand=True)
                if exif[orientation] == 8:
                    I = I.rotate(90,expand=True)


        annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
        anns_sel = coco.loadAnns(annIds)

        target = []
    
        # Show annotations
        for ann in anns_sel:
            #[x, y, w, h]
            bbox = ann['bbox']
            cat = ann['category_id']
            super_cat = self.cat_ids_2_supercat_ids[cat]
            target.append([bbox[0], bbox[1], bbox[2], bbox[3], super_cat])


        target = np.array(target)

        I, target = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(I), np.copy(target))
        # normalize
        X = torch.from_numpy(I).permute(2,0,1)
        X = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(X)

        Y = torch.from_numpy(target)
        return X, Y
    
def get_dataloaders_WASTE(batch_size, num_workers=8, seed=42, data_path='/dtu/datasets1/02514/data_wastedetection'):
    
    
    dataset = WASTE(data_path=data_path)

    generator1 = torch.Generator().manual_seed(seed)
    trainset, valset, testset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    path = "/u/data/s194333/DLCV/Project4_02514-/data"
    dataset = WASTE(data_path=path)
    cats = dataset.super_cat_ids


    train_loader, val_loader, test_loader = get_dataloaders_WASTE(batch_size=1, num_workers=0, seed=42, data_path=path)
    img, target = next(iter(test_loader))

    fig,ax = plt.subplots(1)
    plt.axis('off')
    ax.imshow(img.squeeze(0).permute(1,2,0))
    for box in target[0]:
        rect = Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        # get keyword argument from super category id
        label = list(cats.keys())[list(cats.values()).index(box[4].item())]
        plt.text(box[0]+box[2],box[1],str(label),color='r')
    fig.savefig('test.png', bbox_inches='tight', pad_inches=0)
        



