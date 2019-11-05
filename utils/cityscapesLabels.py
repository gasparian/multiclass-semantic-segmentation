import numpy as np
import pandas as pd

class LabelEncoder:

    def __init__(self):
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
        self.categories = np.arange(self.cityscapes_labels_df["catId"].nunique())

    def classes2cats(self, labelIds):
        """
        converts cityscapes classes to the categories
        in order to decrease the number of classes
        (img --> img)
        """
        for unique in np.unique(labelIds):
            labelIds[labelIds == unique] = self.cityscapes_labels_df[self.cityscapes_labels_df["id"] == unique]["catId"]
        return labelIds.astype(np.uint8)

    def make_ohe(self, labelIds):
        """
        converts image with labels into the one-hot encoded format
        (img[...,] --> img[..., N_CLASSES])
        """    
        ohe_labels = np.zeros(labelIds.shape[:2] + (len(self.categories),))
        for cat in self.categories:
            ys, xs = np.where(labelIds[..., 0] == cat)
            ohe_labels[ys, xs, cat] = 1
        return ohe_labels.astype(np.uint8)

    def inverse_ohe(self, ohe_labels):
        """converts one-hot encoded mask to the 3-ch multiclass mask"""
        inverse_ohe_img = np.zeros(ohe_labels.shape[:2]+(1,))
        for ch in range(ohe_labels.shape[-1]):
            ys, xs = np.where(ohe_labels[..., ch])
            inverse_ohe_img[ys, xs] = ch
        inverse_ohe_img = np.repeat(inverse_ohe_img, 3, axis=2).astype(np.uint8)
        return inverse_ohe_img

    def class2color(self, ohe_labels, mode="catId"):
        """
        converts multiclass mask to (R,G,B) color mask
        mode : `catId` or `classId`
        """
        colored_labels = np.zeros(ohe_labels.shape[:2] + (3,)).astype(np.uint8)
        mode = "catId"

        col = mode
        if mode == "classId":
            col = "id"

        for ch in range(ohe_labels.shape[-1]):
            color = self.cityscapes_labels_df[self.cityscapes_labels_df[col] == ch]["color"].iloc[0]
            ys, xs = np.where(ohe_labels[..., ch])
            colored_labels[ys, xs, :] = color
        return colored_labels


