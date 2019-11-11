import os
import time
import shutil
import random
import warnings

from tqdm import tqdm

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

from utils import Meter, UnetResNet, FPN, TTAWrapper, \
                  LabelEncoder, TrainDataset, CityscapesDataset, post_process

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

MODEL_PATH = "/samsung_drive/semantic_segmentation/tests/best_model.pth"
EVAL_IMAGES_PATH = "/samsung_drive/semantic_segmentation/tests/eval_images"

MODE = "UNET" # UNET or FPN
DEVICE = "cuda:0" # cpu or cuda
SIZE = (512, 1024)
RESIZE = True
TRAIN_ON_CATS = True
APPLY_TTA=False
ACTIVATE=False
BASE_THRESHOLD=0.

if __name__ == "__main__":

    global_start = time.time()
    label_encoder = LabelEncoder()
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
    image_dataset.set_phase("val", valset)
    dataloader = DataLoader(
        image_dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        shuffle=True,   
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

    device = torch.device(DEVICE)
    model.to(device)
    model.eval()
    state = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    if APPLY_TTA:
        TTAModel = TTAWrapper(model, 
                              merge_mode="mean", 
                              activate=ACTIVATE, 
                              temperature=0.5)

    meter = Meter(base_threshold=BASE_THRESHOLD)

    try:
        shutil.rmtree(EVAL_IMAGES_PATH)
    except:
        pass
    os.mkdir(EVAL_IMAGES_PATH)

    start = time.time()
    for batch in tqdm(dataloader):
        images, targets, image_id = batch

        images = images.to(device)
        if APPLY_TTA:
            outputs = TTAModel(images)
        else:
            outputs = model(images)
            if ACTIVATE:
                outputs = torch.sigmoid(outputs)
        if RESIZE:
            outputs = torch.nn.functional.interpolate(outputs, size=(1024, 2048), mode='bilinear', align_corners=True)

        outputs = outputs.detach().cpu()
        meter.update("val", targets, outputs)

        # dump predictions as images
        outputs = (outputs > BASE_THRESHOLD).int() # thresholding
        outputs = outputs.squeeze().permute(1, 2, 0).numpy()
        pic = label_encoder.class2color(outputs, mode="catId")
        pred_name = "_".join(image_id[0].split("_")[:-1]) + "_predicted_mask.png"
        cv2.imwrite(os.path.join(EVAL_IMAGES_PATH, pred_name), pic)

    torch.cuda.empty_cache()
    dices, iou = meter.get_metrics("val")

    print("***** Prediction done in {} sec.; IoU: {}, Dice: {} ***** \n(total elapsed time: {} sec.) ".\
            format(int(time.time()-start), iou, dices[0], int(time.time()-global_start)))
