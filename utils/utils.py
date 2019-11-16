import os
import yaml

import cv2
import numpy as np

import torchvision

import warnings
warnings.filterwarnings("ignore")

def open_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

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

def invert_mask(mask):
    return np.bitwise_not(mask.astype(bool)).astype(int)

class DropClusters:
    '''
    Post processing of each predicted mask, 
    components with lesser number of pixels 
    than `min_size` are ignored
    '''

    @classmethod
    def drop(self, mask, min_size=50*50):
        self.min_size = min_size
        for i in range(2):
            mask = self.filt_invert(mask)
        return mask

    @classmethod
    def filt_invert(self, mask):
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
        predictions = np.zeros(mask.shape[:2], np.int)
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > self.min_size:
                predictions[p] = 1
        inverse_mask = invert_mask(predictions)
        return inverse_mask

def load_train_config(path="train_config.yaml"):
    with open(path) as f:
        data = yaml.load(f)
    return data

def torch2np(outputs):
    outputs = outputs.squeeze(0).permute(1, 2, 0).numpy()
    return outputs
