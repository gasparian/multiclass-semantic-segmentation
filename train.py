import os
import warnings
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import TrainDataset, LabelEncoder, CityscapesDataset, \
                  Trainer, Meter, UnetResNet, FPN 

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# for the TrainDataset initialization
path_masks = "./data/cityscapes/gtFine/gtFine"
path_img = "./data/cityscapes/leftImg8bit"
train_root_path = "./data/cityscapes/leftImg8bit/train"

MODE = "UNET" # UNET or FPN
SIZE = (512, 1024)
TRAIN_ON_CATS = True

if __name__ == "__main__":

    train_dataset = TrainDataset(
        path_masks=path_masks, 
        path_img=path_img, 
        train_root_path=train_root_path, 
        val_cities=["ulm", "bremen", "aachen"]
    )

    trainset, valset = train_dataset.get_paths()

    image_dataset = CityscapesDataset(
        hard_augs=False, resize=SIZE, train_on_cats=TRAIN_ON_CATS
    )

    if MODE == "UNET":
        model = UnetResNet(encoder_name="resnext50", 
                        num_classes=8, 
                        input_channels=3, 
                        num_filters=32, 
                        Dropout=.2, 
                        res_blocks_dec=False)
    elif MODE == "FPN":
        model = FPN(encoder_name='resnext50',
                    decoder_pyramid_channels=256,
                    decoder_segmentation_channels=128,
                    classes=8,
                    dropout=0.2,
                    activation='sigmoid',
                    final_upsampling=4,
                    decoder_merge_policy='add')

    model_trainer = Trainer(model, 
                            "/samsung_drive/semantic_segmentation/UNet", 
                            image_dataset, reset=True,
                            
                            weights_decay=0.0,
                            batch_size=6,
                            accumulation_batches=1,
                            freeze_n_iters=0, 
                            lr=1e-3, 
                            num_epochs=30, 
                            devices_ids=[0, 1],
                            key_metric="dice",
                            
    #                         base_threshold=.5,
    #                         activate=True,
                            base_threshold=.0,
                            activate=False,
                            
    #                         bce_loss_weight=.7,
                            bce_loss_weight=1.,
                            class_weights=[1/8 for i in range(8)], 
                            optimizer=optim.Adam,
                            scheduler_patience=3)

    model_trainer.start(trainset, valset)