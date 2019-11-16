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

def drop_clusters(masks, min_size=50*50):
    '''
    Post processing of each predicted mask, 
    components with lesser number of pixels 
    than `min_size` are ignored
    '''
    if masks.ndim < 3:
        masks = masks[..., np.newaxis]
    predictions = np.zeros(masks.shape, np.float32)
    for ch in range(masks.shape[-1]):
        mask = masks[..., ch]
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
        prediction = np.zeros(mask.shape, np.float32)
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                prediction[p] = 1
        predictions[..., ch] = prediction
    return predictions

def load_train_config(path="train_config.yaml"):
    with open(path) as f:
        data = yaml.load(f)
    return data
    