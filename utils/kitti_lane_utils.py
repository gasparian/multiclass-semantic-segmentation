import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from albumentations import HorizontalFlip, RandomCrop, Resize, \
                           RandomBrightness, RandomContrast, RandomGamma, \
                           RandomFog, RandomRain, RandomShadow, RandomSnow, RandomSunFlare, \
                           Normalize, Resize, Compose, OneOf
from albumentations.pytorch import ToTensor

from .utils import open_img
from .cityscapes_utils import CityscapesDataset

class TrainDataset:

    """
    returns paths for train and validation annotated pairs
    """

    def __init__(self, path_masks, path_img, val_frac=.2, mode="lane"):
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

            pair = (
                os.path.join(self.path_img, fname), 
                os.path.join(self.path_masks, name)
            )
            
            if i < train_len:
                train_dataset.append(pair)
            else:
                val_dataset.append(pair)
                
        return train_dataset, val_dataset


class KittiLaneLabelEncoder:

    def __init__(self):
        raise NotImplementedError

    def make_ohe(self, labelIds):
        raise NotImplementedError

    def inverse_ohe(self, ohe_labels):
        raise NotImplementedError

    def class2color(self, ohe_labels):
        raise NotImplementedError

class KittiLaneDataset(CityscapesDataset):

    def __init__(self, hard_augs=False, resize=None, select_classes=[], orig_size=(1024, 2048)):
        super().__init__(hard_augs, resize, select_classes, orig_size)
        self.label_encoder = KittiLaneLabelEncoder()

    def __getitem__(self, idx):
        image_id = self.data_set[idx]
        img = open_img(image_id[0])
        if self.phase != "test":
            labelIds = open_img(image_id[1])
            mask = self.label_encoder.make_ohe(labelIds)
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
            mask = None
        return img, mask, image_id[0].split("/")[-1]
