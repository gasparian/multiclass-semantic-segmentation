import os

import numpy as np
import pandas as pd

from .utils import open_img

class TrainDataset:

    """
    returns paths for train and validation annotated pairs;
    cities for validation has been chosen to satisfy ~20% limitfor validation;  
    """

    def __init__(self, path_masks, path_img, train_root_path, val_cities=["ulm", "bremen", "aachen"]):
        self.path_masks = path_masks
        self.path_img = path_img
        self.train_root_path = train_root_path
        self.val_cities = val_cities

    def get_paths(self):
        
        train_cities = [city for city in os.listdir(self.train_root_path) if city not in self.val_cities]
        train_dataset, val_dataset = [], []

        for cities, datasets in zip([train_cities, self.val_cities], [train_dataset, val_dataset]):
            all_names = []
            for city in cities:
                names = os.listdir(os.path.join(self.train_root_path, city))
                names = set([os.path.join(city, "_".join(n.split("_")[:3])) for n in names])
                all_names += names
            
            for name in all_names:
                datasets.append((
                    os.path.join(self.path_img, "train", name+"_leftImg8bit.png"), 
                    os.path.join(self.path_masks, "train", name+"_gtFine_labelIds.png")
                ))
                
        np.random.shuffle(train_dataset)
        np.random.shuffle(val_dataset)
        return train_dataset, val_dataset

class CityscapesDataset(Dataset):
    
    def __init__(self, hard_augs=False, resize=None, train_on_cats=True, select_classes=[]):
        self.orig_h, self.orig_w = 1024, 2048
        self.h, self.w = self.orig_h, self.orig_w
        self.resize = resize
        if self.resize is not None:
            self.h, self.w = self.resize
        self.data_set = []
        self.phase = "train"
        self.hard_augs = hard_augs
        self.train_on_cats = train_on_cats
        self.label_encoder = LabelEncoder(select_classes)
        # define random crop h and w based 
        # on the original image size: 
        self.random_crop_h = int(self.orig_h*0.9)
        self.random_crop_w = int(self.orig_w*0.9)
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.final_resizing = Resize(self.h, self.w, interpolation=4, p=1.0)
        self.possible_phases = ["train", "val", "test"]

    def set_phase(self, phase, data_set):
        """
        must called after init or to swap training phase 
        """
        self.phase = phase
        if phase not in self.possible_phases:
            raise ValueError('Phase type must be on of: {}'.format(self.possible_phases))
        self.data_set = data_set
        self.transformer = self.get_transforms()
        
    def get_transforms(self):
        list_transforms = []
        if self.phase == "train":
            list_transforms.extend([
                    # Spatial transforms
                    HorizontalFlip(p=0.6),
                    RandomCrop(self.random_crop_h, self.random_crop_w, p=0.8),
                    Resize(self.orig_h, self.orig_w, interpolation=4, p=1.0),
                    # RGB transormations
                    OneOf([
                        RandomBrightness(p=0.5, limit=0.2),
                        RandomContrast(p=0.5, limit=0.2),
                        RandomGamma(p=0.5, gamma_limit=(80, 120))
                    ], p=1.)
                ])
                
            if self.hard_augs:
                list_transforms.extend(
                    # Hard augs 
                    OneOf([
                        RandomFog(p=0.5, fog_coef_lower=0.1, fog_coef_upper=.3, alpha_coef=0.08),
                        RandomRain(p=0.5, slant_lower=-20, slant_upper=20, rain_type=None), # [None, "drizzle", "heavy", "torrestial"]
                        RandomSnow(p=0.5, snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5),
                        RandomSunFlare(p=0.5, flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=3, 
                                        num_flare_circles_upper=6, src_radius=400, src_color=(255, 255, 255))  
                    ], p=0.8)
                )

        list_transforms.append(Normalize(mean=self.mean, std=self.std, p=1))
        list_trfms = Compose(list_transforms)
        return list_trfms

    def __getitem__(self, idx):
        image_id = self.data_set[idx]    
        img = open_img(image_id[0])
        if self.phase != "test":
            labelIds = open_img(image_id[1])
            mask = self.label_encoder.make_ohe(labelIds, mode="catId" if self.train_on_cats else "trainId")
            img, mask = self.transformer(image=img, mask=mask).values()
        else:
            img = self.transformer(image=img)["image"]
        if self.resize is not None:
            img = self.final_resizing(image=img)["image"]
        if self.phase != "test":
            img, mask = ToTensor()(image=img, mask=mask).values()
            mask = mask[0].permute(2, 0, 1) # N_CLASSESxHxW
            return img, mask, image_id[0].split("/")[-1]
        else:
            img = ToTensor()(image=img)["image"]
            return img, None

    def __len__(self):
        return len(self.data_set)
