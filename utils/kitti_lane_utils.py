import re
import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from albumentations import HorizontalFlip, RandomCrop, Resize, \
                           RandomBrightness, RandomContrast, RandomGamma, \
                           RandomFog, RandomRain, RandomShadow, RandomSnow, RandomSunFlare, \
                           Normalize, Resize, Compose, OneOf
from albumentations.pytorch import ToTensor

from .utils import open_img, DropClusters, invert_mask
from .cityscapes_utils import CityscapesDataset

class KittiTrainDataset:

    """
    returns paths for train and validation annotated pairs
    """

    def __init__(self, path_masks, path_img, val_frac=.2, mode="lane", test_root_path=None):
        self.path_masks = path_masks
        self.path_img = path_img
        self.val_frac = val_frac
        possible_modes = ["lane", "road"]
        if mode not in possible_modes:
            raise ValueError('Mode must be on of: {}'.format(possible_modes))
        self.mode = mode

    def get_paths(self):
        train_dataset, val_dataset = [], []
        fnames = os.listdir(self.path_img)
        np.random.shuffle(fnames)
        train_len = int((1 - self.val_frac) * len(fnames))
 
        for i, fname in enumerate(fnames):

            name = fname.split("_")
            name = "_".join([name[0], self.mode, name[1]])

            img_path = os.path.join(self.path_img, fname)
            mask_path = os.path.join(self.path_masks, name)

            # look of the mask exists
            try:
                os.stat(mask_path)
                pair = (img_path, mask_path)
                
                if i < train_len:
                    train_dataset.append(pair)
                else:
                    val_dataset.append(pair)
            except:
                pass
                
        return train_dataset, val_dataset

def KittiTestDataset(test_root_path):

    """
    returns paths for hold-out test images
    """
        
    names = os.listdir(test_root_path)
    dataset = [[os.path.join(test_root_path, name)] for name in names]
            
    return dataset

class KittiLaneLabelEncoder:

    def __init__(self):
        self.label_color = (128, 64, 128) # road; color from Cityscapes color-scheme

    def encode(self, labels):
        """ 3-channel binary mask --> 1-channel binary mask """
        # use 2 channel since BGR2RGB convertion
        labels = labels.astype(float)[..., 2] / 255 
        return labels[..., np.newaxis]

    def class2color(self, labels, clean_up_clusters=0, mode=None):
        """ 1-channel binary mask --> 3-channel image """
        clean_up_clusters *= clean_up_clusters # create an area
        colored_labels = np.zeros(labels.shape[:2] + (3,)).astype(np.uint8)
        labels = np.squeeze(labels)
        if clean_up_clusters > 0:
            labels = DropClusters.drop(labels, min_size=clean_up_clusters)
        ys, xs = np.where(labels)
        colored_labels[ys, xs, :] = self.label_color
        return colored_labels

class KittiLaneDataset(CityscapesDataset):

    def __init__(self, hard_augs=False, resize=None, orig_size=(375, 1242), select_classes=[], train_on_cats=None):

        super().__init__(hard_augs=hard_augs, resize=resize, orig_size=orig_size)
        self.resize_to_orig = Resize(self.orig_h, self.orig_w, interpolation=4, p=1.0)
        self.label_encoder = KittiLaneLabelEncoder()

    def __getitem__(self, idx):
        image_id = self.data_set[idx]
        img = open_img(image_id[0])
        if self.phase != "test":
            labelIds = open_img(image_id[1])
            mask = self.label_encoder.encode(labelIds)
            if mask.shape[:2] != self.orig_size:
                mask = self.resize_to_orig(image=mask)["image"].astype(bool).astype(int)
            img, mask = self.transformer(image=img, mask=mask).values()
        else:
            img = self.transformer(image=img)["image"]
        if self.resize is not None:
            img = self.final_resizing(image=img)["image"]
        if self.phase != "test":
            img, mask = ToTensor()(image=img, mask=mask).values()
            mask = mask[0].permute(2, 0, 1) # N_CLASSESxHxW
        else:
            img = ToTensor()(image=img)["image"]
            mask = img
        return img, mask, image_id[0]
