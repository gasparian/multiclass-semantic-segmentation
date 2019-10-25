import os

import cv2
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import Dataset
import torchvision

from albumentations import HorizontalFlip, VerticalFlip, Normalize, Resize, Compose
from albumentations.pytorch import ToTensor

import warnings
warnings.filterwarnings("ignore")

def get_encoder(model, pretrained=True):
    if model == "resnet18":
        encoder = torchvision.models.resnet18(pretrained=pretrained)
    elif model == "resnet34":
        encoder = torchvision.models.resnet34(pretrained=pretrained)
    elif model == "resnet50":
        encoder = torchvision.models.resnet50(pretrained=pretrained)
    elif model == "resnext50":
        encoder = torchvision.models.resnext50_32x4d(pretrained=pretrained)
    elif model == "resnext101":
        encoder = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        
    if model in ["resnet18", "resnet34"]: 
        model = "resnet18-34"
    else: 
        model = "resnet50-101"
        
    filters_dict = {
        "resnet18-34": [512, 512, 256, 128, 64],
        "resnet50-101": [2048, 2048, 1024, 512, 256]
    }

    return encoder, filters_dict[model]

def open_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
def create_mask(labels):
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)
    for idx, label in enumerate(labels):
        if label is not np.nan:
            mask_label = np.zeros(1600*256, dtype=np.uint8)
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos-1:pos+le-1] = 1
            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')
    return mask    

def get_mask(base_name, train_df):
    # train_df : global variable
    labels = train_df[train_df.ImageId_ClassId.str.contains(base_name)][["EncodedPixels"]].values.flatten()
    return create_mask(labels)

#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    ''' 
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_mask(labels):
    '''Given RLE-encoded mask, returns mask (256, 1600, 4)'''
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return masks

class SteelDataset(Dataset):
    
    def __init__(self, data_set, phase, resize=None):
        self.data_set = data_set
        self.df = pd.read_csv(os.path.join("./data", "train.csv"))
        self.phase = phase
        self.resize = resize
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.transforms = self.get_transforms()
        
    def get_transforms(self):
        list_transforms = []
        if self.phase == "train":
            list_transforms.extend(
                [
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                ]
            )
            
        list_transforms.extend(
            [
                Normalize(mean=self.mean, std=self.std, p=1),
                ToTensor(),
            ]
        )
        list_trfms = Compose(list_transforms)
        return list_trfms

    def __getitem__(self, idx):
        image_id = self.data_set[idx]
        labels = self.df[self.df.ImageId_ClassId.str.contains(image_id)].\
                    sort_values("ImageId_ClassId")["EncodedPixels"].values
        mask = make_mask(labels)
        image_path = os.path.join("./data", "train_images",  image_id)
        img = cv2.imread(image_path)
        
        if self.resize is not None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)
        
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image'] # 3x256x1600
        mask = augmented['mask'] # 256x1600x4
        mask = mask[0].permute(2, 0, 1) # 4x256x1600
        return img, mask

    def __len__(self):
        return len(self.data_set)

def post_process(preds, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''

    # mask = cv2.threshold(preds, threshold, 1, cv2.THRESH_BINARY)[1]
    # mask = mask.astype(np.uint8)

    mask = (preds > threshold).astype(np.uint8)

    num_component, component = cv2.connectedComponents(mask)
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num