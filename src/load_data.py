import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
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

class WASTE(torch.utils.data.Dataset):
    def __init__(self, transform, data_path='/dtu/datasets1/02514/data_wastedetection'):
        'Initialization'
        self.path = sorted(glob.glob(data_path + '/annotations.json'))
        with open(self.path, 'r') as f:
            dataset = json.loads(f.read())

        self.categories = dataset['categories']
        anns = dataset['annotations']
        imgs = dataset['images']

        self.img_ids = []
        for img in imgs:
            self.img_ids.append(img['file_name'])
        
        self.transform = transform
        self.image_path = sorted(glob.glob(data_path))
        
    """
    def __len__(self):
        'Returns the total number of samples'
        return len(self.path)
    """

    def __getitem__(self, idx):
        'Generates one sample of data'
        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        # Loads dataset as a coco object
        coco = COCO(self.path)

        # Load images
        img_id = self.img_ids[idx]
        I = Image.open(self.image_path + '/' + img_id)
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

        # Load mask ids
        annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
        anns_sel = coco.loadAnns(annIds)
    
        bboxes = []
        # Show annotations
        for ann in anns_sel:
            #[x, y, w, h]
            bboxes.append(ann['bbox'])

        transformed = self.transform(image=I, annotation=bboxes)
        X = transformed["image"]
        Y = transformed["annotation"]
        return X, Y
    
def get_dataloaders_WASTE(batch_size, num_workers=8, seed=42, data_path='/dtu/datasets1/02514/data_wastedetection'):
    data_transform_val = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ],
        additional_targets={'annotation'}
    )
    data_transform_train = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ],
        additional_targets={'annotation'}
    )
    
    trainset = WASTE(transform=data_transform_train, data_path=data_path)

    generator1 = torch.Generator().manual_seed(seed)
    trainset, _, _ = random_split(trainset, [0.8, 0.1, 0.1], generator=generator1)
    generator1 = torch.Generator().manual_seed(seed)
    _, valset, testset = random_split(testset, [0.8, 0.1, 0.1], generator=generator1)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader