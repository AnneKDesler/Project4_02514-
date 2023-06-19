import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import numpy as np
import PIL.Image as Image
import json
from PIL import Image, ExifTags
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import data_augment as dataAug

import os
import cv2

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class WASTE(torch.utils.data.Dataset):
    def __init__(self, mode = "train", data_path='/dtu/datasets1/02514/data_wastedetection', img_size=512):
        'Initialization'
        self.img_size = img_size
        self.path = data_path
        self.ann_path = data_path + '/annotations.json'

        with open(self.ann_path, 'r') as f:
            dataset = json.loads(f.read())

        categories = dataset['categories']
        self.anns = dataset['annotations']
        self.imgs = dataset['images']
        img_ids = []
        for img in self.imgs:
            img_ids.append(img['id'])
        
        if mode == "train":
            self.img_ids = img_ids[:int(len(img_ids)*0.8)]
        elif mode == "val":
            self.img_ids = img_ids[int(len(img_ids)*0.8):int(len(img_ids)*0.9)]
        elif mode == "test":
            self.img_ids = img_ids[int(len(img_ids)*0.9):]

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
        blockPrint()
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
        enablePrint()

        target = []
    
        # Show annotations
        for ann in anns_sel:
            #[x, y, w, h] lower left corner and width and height
            # changed to center x, center y, width, height
            bbox = ann['bbox']
            bbox[0] = bbox[0] + bbox[2]/2
            bbox[1] = bbox[1] + bbox[3]/2
            cat = ann['category_id']
            super_cat = self.cat_ids_2_supercat_ids[cat]
            target.append([bbox[0], bbox[1], bbox[2], bbox[3], super_cat])

        I = np.array(I).astype(np.float32)
        I = I/255.
        target = np.array(target)

        I, target = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(I), np.copy(target))
        # normalize
        I = torch.from_numpy(I).permute(2,0,1)
        #I = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(I)
        target = torch.from_numpy(target)
        return I, target, img_id
    
def get_dataloaders_WASTE(batch_size, num_workers=8, seed=42, data_path='/dtu/datasets1/02514/data_wastedetection'):
    trainset = WASTE2(data_path=data_path, mode="train")
    valset = WASTE2(data_path=data_path, mode="val")
    testset = WASTE2(data_path=data_path, mode="test")

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


class WASTE2(torch.utils.data.Dataset):
    def __init__(self, mode = "train", data_path='/dtu/datasets1/02514/data_wastedetection', img_size=512):
        'Initialization'
        self.img_size = img_size
        self.path = data_path
        self.ann_path = data_path + '/annotations.json'

        with open(self.ann_path, 'r') as f:
            dataset = json.loads(f.read())

        categories = dataset['categories']
        self.anns = dataset['annotations']
        self.imgs = dataset['images']
        img_ids = []
        for img in self.imgs:
            img_ids.append(img['id'])

        if mode == "train":
            self.img_ids = img_ids[:int(len(img_ids)*0.8)]
        elif mode == "val":
            self.img_ids = img_ids[int(len(img_ids)*0.8):int(len(img_ids)*0.9)]
        elif mode == "test":
            self.img_ids = img_ids[int(len(img_ids)*0.9):]

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
        blockPrint()
        coco = COCO(self.ann_path)

        # Find image id
        img_id = idx

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
        enablePrint()

        target = []
    
        # Show annotations
        for ann in anns_sel:
            #[x, y, w, h] lower left corner and width and height
            # changed to center x, center y, width, height
            bbox = ann['bbox']
            bbox[0] = bbox[0] + bbox[2]/2
            bbox[1] = bbox[1] + bbox[3]/2
            cat = ann['category_id']
            super_cat = self.cat_ids_2_supercat_ids[cat]
            target.append([bbox[0], bbox[1], bbox[2], bbox[3], super_cat])

        I = np.array(I).astype(np.float32)
        I = I/255.
        target = np.array(target)

        I, target = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(I), np.copy(target))

        return I, target, img_id


class Proposals(torch.utils.data.Dataset):
    def __init__(self, data_path='/dtu/datasets1/02514/data_wastedetection', proposal_path = 'region_proposals2', img_size=512, size=64):
        'Initialization'
        self.size = size
        self.path = data_path
        self.proposal_path = proposal_path

        self.proposal_files = os.listdir(proposal_path)

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
        return len(self.proposal_files)
    

    def __getitem__(self, idx):
        'Generates one sample of data'
        proposal = np.load(self.proposal_path + '/' + self.proposal_files[idx])
        img_id = int(proposal[5])
        rect = proposal[:5]
        x, y, w, h = rect[:4]


        'Generates one sample of data'


        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        # Loads dataset as a coco object
        blockPrint()
        coco = COCO(self.ann_path)

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
        enablePrint()

        target = []
    
        # Show annotations
        for ann in anns_sel:
            #[x, y, w, h] lower left corner and width and height
            # changed to center x, center y, width, height
            bbox = ann['bbox']
            bbox[0] = bbox[0] + bbox[2]/2
            bbox[1] = bbox[1] + bbox[3]/2
            cat = ann['category_id']
            super_cat = self.cat_ids_2_supercat_ids[cat]
            target.append([bbox[0], bbox[1], bbox[2], bbox[3], super_cat])

        I = np.array(I).astype(np.float32)
        I = I/255.
        target = np.array(target)

        I, target = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(I), np.copy(target))

    
        crop = I[y-h//2:y+h//2, x-w//2:x+w//2,:]

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        crop = cv2.resize(crop, (self.size, self.size))
        crop = crop[:, :, (2, 1, 0)]

        #dataAug.Resize((self.size, self.size),False)(np.copy(crop), padding=True)
        I = torch.from_numpy(I).permute(2,0,1)
        crop = torch.from_numpy(crop).permute(2,0,1)
        target = torch.from_numpy(target)
        rect = torch.from_numpy(rect)
        #print(I.shape, crop.shape, target.shape, rect.shape, img_id)
        #return I, target, crop, rect, img_id
        return crop, rect, img_id
    

class Proposals2(torch.utils.data.Dataset):
    def __init__(self, data_path='/dtu/datasets1/02514/data_wastedetection', proposal_path = 'region_proposals2', size=64):
        'Initialization'
        self.size = size
        self.path = data_path
        self.proposal_path = proposal_path
        self.waste2 = WASTE2(data_path=data_path)
        self.proposal_files = os.listdir(proposal_path)

    
    def __len__(self):
        'Returns the total number of samples'
        return len(self.proposal_files)
    

    def __getitem__(self, idx):
        'Generates one sample of data'
        proposal = np.load(self.proposal_path + '/' + self.proposal_files[idx])
        img_id = int(proposal[5])
        rect = proposal[:5]
        x, y, w, h = rect[:4]


        I, target, _ = self.waste2.__getitem__(img_id)
    
        crop = I[y-h//2:y+h//2, x-w//2:x+w//2,:]

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        crop = cv2.resize(crop, (self.size, self.size))
        crop = crop[:, :, (2, 1, 0)]

        #dataAug.Resize((self.size, self.size),False)(np.copy(crop), padding=True)
        I = torch.from_numpy(I).permute(2,0,1)
        crop = torch.from_numpy(crop).permute(2,0,1)
        target = torch.from_numpy(target)
        rect = torch.from_numpy(rect)

        return I, target, crop, rect, img_id


class Proposals3(torch.utils.data.Dataset):
    def __init__(self, mode = "train", data_path='/dtu/datasets1/02514/data_wastedetection', proposal_path = 'region_proposals2', size=64):
        'Initialization'
        self.size = size
        if mode == "train":
            self.proposal_path = proposal_path + "/train"
        elif mode == "val":
            self.proposal_path = proposal_path + "/val"
        elif mode == "test":
            self.proposal_path = proposal_path + "/test"

        
        self.waste2 = WASTE2(data_path=data_path, mode=mode)
        self.proposal_files = os.listdir(self.proposal_path)

    
    def __len__(self):
        'Returns the total number of samples'
        return len(self.proposal_files)
    

    def __getitem__(self, idx):
        'Generates one sample of data'

        proposal = np.load(self.proposal_path + '/' + self.proposal_files[idx])
        img_id = int(proposal[5])
        rect = proposal[:5]

        x, y, w, h = rect[:4]

        I, target, _ = self.waste2.__getitem__(img_id)
    
        crop = I[y-h//2:y+h//2, x-w//2:x+w//2,:]

        try:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            crop = cv2.resize(crop, (self.size, self.size))
        except:
            crop = np.zeros((self.size, self.size, 3)).astype(np.float32)
            rect[4] = 28
        crop = crop[:, :, (2, 1, 0)]


        #dataAug.Resize((self.size, self.size),False)(np.copy(crop), padding=True)

        crop = torch.from_numpy(crop).permute(2,0,1)
        target = torch.from_numpy(target)
        rect = torch.from_numpy(rect)

        return crop, rect, img_id



def get_dataloaders_proposals(batch_size, num_workers=8, seed=42, data_path='/dtu/datasets1/02514/data_wastedetection', proposal_path = 'region_proposals4', size=32):
    
    trainset = Proposals3(data_path=data_path, proposal_path=proposal_path, mode="train")
    valset = Proposals3(data_path=data_path,proposal_path=proposal_path, mode="val")
    testset = Proposals3(data_path=data_path,proposal_path=proposal_path, mode="test")

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    path = "/u/data/s194333/DLCV/Project4_02514-/data"
    dataset = WASTE2(data_path=path)
    cats = dataset.super_cat_ids

    train_loader, val_loader, test_loader = get_dataloaders_WASTE(batch_size=1, num_workers=0, seed=42, data_path=path)
    img, target, img_id = next(iter(test_loader))

    fig,ax = plt.subplots(1)
    plt.axis('off')
    ax.imshow(img.squeeze(0))#.permute(1,2,0))
    for box in target[0]:
        rect = Rectangle((box[0]-box[2]/2,box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        # get keyword argument from super category id
        label = list(cats.keys())[list(cats.values()).index(box[4].item())]
        plt.text(box[0]+box[2]/2,box[1]-box[3]/2,str(label),color='r')

    fig.savefig(str(img_id.item())+'.png', bbox_inches='tight', pad_inches=0)
    
    
    trainloader, valloader, testloader = get_dataloaders_proposals(batch_size=1, num_workers=0, seed=42, data_path=path, proposal_path = 'region_proposals4', size=64)
    prop, rect, img_id = next(iter(trainloader))
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
    
    box = rect[0] 
    gt = Rectangle((box[0]-box[2]/2,box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(gt)
    # get keyword argument from super category id
    if box[4].item() != 28:
        label = list(cats.keys())[list(cats.values()).index(box[4].item())]
    else:
        label = 'background'
    plt.text(box[0]+box[2]/2,box[1]-box[3]/2,str(label),color='b')
    
    fig.savefig(str(img_id.item())+'.png', bbox_inches='tight', pad_inches=0)

    fig,ax = plt.subplots(1)
    plt.axis('off')
    ax.imshow(prop.squeeze(0).permute(1,2,0))
    fig.savefig(str(img_id.item())+'_prop.png', bbox_inches='tight', pad_inches=0)

    
