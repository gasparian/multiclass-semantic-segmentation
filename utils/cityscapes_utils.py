import os

import numpy as np
import pandas as pd

from albumentations.pytorch import ToTensor

from .utils import open_img, DropClusters, invert_mask
from dataset import CustomDataset

class CityscapesLabelEncoder:

    def __init__(self, select_classes=[]):
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py  
        self.labels = [
            (                   "name","id", "trainId",         "category",  "catId","hasInstances","ignoreInEval",        "color"),
            (  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
            (  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
            (  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
            (  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
            (  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
            (  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
            (  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
            (  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
            (  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
            (  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
            (  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
            (  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
            (  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
            (  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
            (  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
            (  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
            (  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
            (  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
            (  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
            (  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
            (  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
            (  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
            (  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
            (  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
            (  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
            (  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
            (  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
            (  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
            (  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
            (  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
        ]

        # create labels dataframe
        self.cityscapes_labels_df = pd.DataFrame(self.labels[1:], columns=self.labels[0])
        self.cityscapes_labels_df.loc[self.cityscapes_labels_df["trainId"].isin([255, -1]), "trainId"] = 19
        self.categories = np.arange(self.cityscapes_labels_df["catId"].nunique())
        if select_classes:
            selected = self.cityscapes_labels_df[
                self.cityscapes_labels_df["name"].isin(select_classes)]["id"].unique()
            self.cityscapes_labels_df.loc[~self.cityscapes_labels_df["id"].isin(selected), "trainId"] = len(selected)
            for i, j in enumerate(selected):
                self.cityscapes_labels_df.loc[self.cityscapes_labels_df["id"] == j, "trainId"] = i
        self.classes = self.cityscapes_labels_df["trainId"].unique()
        self.classes.sort() # in-place labels ascending sort

    def make_ohe(self, labelIds, mode="catId"):
        """
        converts image with labels into the one-hot encoded format
        (img[...,] --> img[..., N_CLASSES])
        mode : `catId` or `trainId`
        """ 
        classes = self.categories
        if mode == "trainId":
            classes = self.classes

        if len(classes) == 2:
            classes = [0]

        for unique in np.unique(labelIds):
            labelIds[labelIds == unique] = self.cityscapes_labels_df[self.cityscapes_labels_df["id"] == unique][mode]
        labelIds = labelIds.astype(int)

        ohe_labels = np.zeros(labelIds.shape[:2] + (len(classes),))
        for c in classes:
            ys, xs = np.where(labelIds[..., 0] == c)
            ohe_labels[ys, xs, c] = 1
        return ohe_labels.astype(int)

    def inverse_ohe(self, ohe_labels):
        """converts one-hot encoded mask to the multiclass mask"""
        inverse_ohe_img = np.zeros(ohe_labels.shape[:2]+(1,))
        for ch in range(ohe_labels.shape[-1]):
            ys, xs = np.where(ohe_labels[..., ch])
            inverse_ohe_img[ys, xs] = ch
        inverse_ohe_img = np.repeat(inverse_ohe_img, 3, axis=2).astype(int)
        return inverse_ohe_img

    def class2color(self, ohe_labels, mode="catId", clean_up_clusters=0):
        """
        converts multiclass mask to (R,G,B) color mask
        mode : `catId` or `trainId`
        """
        clean_up_clusters *= clean_up_clusters # create an area
        if ohe_labels.shape[-1] == 1:
            ohe_labels = np.concatenate([ohe_labels, invert_mask(ohe_labels)], axis=2)

        colored_labels = np.zeros(ohe_labels.shape[:2] + (3,)).astype(np.uint8)
        for ch in range(ohe_labels.shape[-1]):
            color = self.cityscapes_labels_df[self.cityscapes_labels_df[mode] == ch]["color"].iloc[0]
            if clean_up_clusters > 0:
                ohe_labels[..., ch] = DropClusters.drop(ohe_labels[..., ch], min_size=clean_up_clusters)
            ys, xs = np.where(ohe_labels[..., ch])
            colored_labels[ys, xs, :] = color
        return colored_labels

class CityscapesTrainDataset:

    """
    returns paths for train and validation annotated pairs;
    cities for validation has been chosen to satisfy ~20% limitfor validation;  
    """

    def __init__(self, path_masks, path_img, train_root_path,
                 val_cities=["ulm", "bremen", "aachen"], test_root_path=None):

        self.path_masks = path_masks
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
                    os.path.join(self.train_root_path, name+"_leftImg8bit.png"), 
                    os.path.join(self.path_masks, "train", name+"_gtFine_labelIds.png")
                ))
                
        np.random.shuffle(train_dataset)
        np.random.shuffle(val_dataset)
        return train_dataset, val_dataset

def CityscapesTestDataset(test_root_path):

    """
    returns paths for hold-out test images
    """
        
    cities = os.listdir(test_root_path)
    dataset = []

    for city in cities:
        names = os.listdir(os.path.join(test_root_path, city))        
        dataset.extend([[os.path.join(test_root_path, city, name)] for name in names])
            
    return dataset

class CityscapesDataset(CustomDataset):
    
    def __init__(self, hard_augs=False, resize=None, train_on_cats=True, select_classes=[], orig_size=None):
        super().__init__(hard_augs=hard_augs, resize=resize, orig_size=orig_size)
        self.label_encoder = CityscapesLabelEncoder(select_classes)
        self.train_on_cats = train_on_cats

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
        else:
            img = ToTensor()(image=img)["image"]
            mask = img
        return img, mask, image_id[0]
