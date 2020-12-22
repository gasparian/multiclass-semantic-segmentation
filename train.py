import os
import random
import warnings
import argparse
from shutil import copyfile

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import CityscapesTrainDataset, CityscapesLabelEncoder, CityscapesDataset, \
                  KittiLaneLabelEncoder, KittiTrainDataset, KittiLaneDataset, \
                  Trainer, Meter, UnetResNet, FPN, load_train_config

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()
config = load_train_config(args.config_path)
globals().update(config)

if __name__ == "__main__":

    if TARGET == "kitti":
        train_dataset = KittiTrainDataset(**PATHS["KITTI"])
        trainset, valset = train_dataset.get_paths()
        image_dataset = KittiLaneDataset(**DATASET)
    
    elif TARGET == "cityscapes":
        train_dataset = CityscapesTrainDataset(**PATHS["CITYSCAPES"])
        trainset, valset = train_dataset.get_paths()
        image_dataset = CityscapesDataset(**DATASET)

    if MODEL["mode"] == "UNET":
        model = UnetResNet(encoder_name=MODEL["backbone"], 
                           num_classes=MODEL["num_classes"], 
                           input_channels=3, 
                           num_filters=32, 
                           Dropout=0.3, 
                           res_blocks_dec=MODEL["unet_res_blocks_decoder"])

    elif MODEL["mode"] == "FPN":
        model = FPN(encoder_name=MODEL["backbone"],
                    decoder_pyramid_channels=256,
                    decoder_segmentation_channels=128,
                    classes=MODEL["num_classes"],
                    dropout=0.3,
                    activation='sigmoid',
                    final_upsampling=4,
                    decoder_merge_policy='add')
    else:
        raise ValueError('Model type is not correct: `{}`.'.format(MODEL["mode"]))

    model_trainer = Trainer(model=model, image_dataset=image_dataset, optimizer=optim.Adam, **TRAINING)
    model_trainer.start(trainset, valset)

    # copy training config file into created folder
    copyfile(args.config_path, os.path.join(TRAINING["model_path"], "train_config.yaml"))
