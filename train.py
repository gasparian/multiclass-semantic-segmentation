import os
import warnings
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import SteelDataset
from utils.trainer import Trainer
from utils.Unet import UnetResNet
from utils.FPN import FPN

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_masks_area = pd.read_csv("./data/train_masks_area.csv", dtype=object, index_col=None)
train_imgs, val_imgs = \
    train_test_split(train_masks_area["ImageId"].values, test_size=.2, stratify=train_masks_area["area_hash"], 
                         shuffle=True, random_state=42)

model = UnetResNet(encoder_name="resnext50", 
                   num_classes=4, 
                   input_channels=3, 
                   avg_pool_kernel=(8, 50),
                   num_filters=32, 
                   Dropout=.2, 
                   res_blocks_dec=False)

# model = FPN(encoder_name='resnext50',
#             decoder_pyramid_channels=256,
#             decoder_segmentation_channels=128,
#             classes=4,
#             dropout=0.2,
#             activation='sigmoid',
#             final_upsampling=4,
#             decoder_merge_policy='add')

model_trainer = Trainer(model, "tests", SteelDataset, reset=True,
                        weights_decay=0.0,
                        batch_size=8,
                        accumulation_batches=1,
                        freeze_n_iters=0, 
                        lr=5e-4, 
                        resize=None,
                        num_epochs=30, 
                        devices_ids=[0, 1],
                        key_metric="dice",
                        
                        # base_threshold=.5,
                        # activate=True,
                        # bce_loss_weight=1.,
                        base_threshold=.0,
                        activate=False,
                        bce_loss_weight=.7,
                        class_weights=[1/4 for i in range(4)], 
                        
                        optimizer=optim.Adam,
                        scheduler_patience=3)

model_trainer.start(train_imgs, val_imgs)