import os
import time
import shutil
import random
import argparse
import warnings

from tqdm import tqdm

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

from utils import Meter, UnetResNet, FPN, TTAWrapper, load_train_config, \
                  LabelEncoder, TrainDataset, CityscapesDataset, post_process

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

    global_start = time.time()

    label_encoder = LabelEncoder()
    train_dataset = TrainDataset(**TRAIN_DATASET)
    trainset, valset = train_dataset.get_paths()
    image_dataset = CityscapesDataset(**CITYSCAPES_DATASET)

    image_dataset.set_phase("val", valset)
    dataloader = DataLoader(
        image_dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        shuffle=True,   
    )

    if MODEL["mode"] == "UNET":
        model = UnetResNet(encoder_name=MODEL["backbone"], 
                           num_classes=MODEL["num_classes"], 
                           input_channels=3, 
                           num_filters=32, 
                           Dropout=0.2, 
                           res_blocks_dec=False)

    elif MODEL["mode"] == "FPN":
        model = FPN(encoder_name=MODEL["backbone"],
                    decoder_pyramid_channels=256,
                    decoder_segmentation_channels=128,
                    classes=MODEL["num_classes"],
                    dropout=0.2,
                    activation='sigmoid',
                    final_upsampling=4,
                    decoder_merge_policy='add')
    else:
        raise ValueError('Model type is not correct: `{}`.'.format(MODEL["mode"]))

    device = torch.device(EVAL["device"])
    model.to(device)
    model.eval()
    state = torch.load(EVAL["model_path"], map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    if APPLY_TTA:
        TTAModel = TTAWrapper(model, 
                              merge_mode="mean", 
                              activate=EVAL["activate"])

    meter = Meter(base_threshold=EVAL["base_threshold"])

    try:
        shutil.rmtree(EVAL["eval_images_path"])
    except:
        pass
    os.mkdir(EVAL["eval_images_path"])

    start = time.time()
    for batch in tqdm(dataloader):
        images, targets, image_id = batch

        images = images.to(device)
        if EVAL["apply_tta"]:
            outputs = TTAModel(images)
        else:
            outputs = model(images)
            if EVAL["activate"]:
                outputs = torch.sigmoid(outputs)
        if EVAL["resize"]:
            outputs = torch.nn.functional.interpolate(outputs, size=(1024, 2048), mode='bilinear', align_corners=True)

        outputs = outputs.detach().cpu()
        meter.update("val", targets, outputs)

        # dump predictions as images
        outputs = (outputs > EVAL["base_threshold"]).int() # thresholding
        outputs = outputs.squeeze().permute(1, 2, 0).numpy()
        pic = label_encoder.class2color(outputs, mode="catId")
        pred_name = "_".join(image_id[0].split("_")[:-1]) + "_predicted_mask.png"
        cv2.imwrite(os.path.join(EVAL["eval_images_path"], pred_name), pic)

    torch.cuda.empty_cache()
    dices, iou = meter.get_metrics("val")

    print("***** Prediction done in {} sec.; IoU: {}, Dice: {} ***** \n(total elapsed time: {} sec.) ".\
            format(int(time.time()-start), iou, dices[0], int(time.time()-global_start)))
