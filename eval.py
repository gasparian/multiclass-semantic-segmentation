import os
import re
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

from utils import Meter, UnetResNet, FPN, TTAWrapper, load_train_config, CityscapesTestDataset, torch2np, \
                  KittiTrainDataset, KittiTestDataset, KittiLaneDataset, \
                  CityscapesTrainDataset, CityscapesDataset, open_img

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

    if not EVAL["test_mode"]:

        if TARGET == "kitti":
            train_dataset = KittiTrainDataset(**PATHS["KITTI"])
            trainset, valset = train_dataset.get_paths()
            image_dataset = KittiLaneDataset(**DATASET)

        elif TARGET == "cityscapes":
            train_dataset = CityscapesTrainDataset(**PATHS["CITYSCAPES"])
            trainset, valset = train_dataset.get_paths()
            image_dataset = CityscapesDataset(**DATASET)

        image_dataset.set_phase("val", valset)

    else:

        if TARGET == "kitti":
            testset = KittiTestDataset(PATHS["KITTI"]["test_root_path"])
            image_dataset = KittiLaneDataset(**DATASET)
        
        elif TARGET == "cityscapes":
            testset = CityscapesTestDataset(PATHS["CITYSCAPES"]["test_root_path"])
            image_dataset = CityscapesDataset(**DATASET)

        image_dataset.set_phase("test", testset)

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
                           res_blocks_dec=MODEL["unet_res_blocks_decoder"])

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

    if EVAL["apply_tta"]:
        TTAModel = TTAWrapper(model, merge_mode="mean")

    if not EVAL["test_mode"]:
        meter = Meter(base_threshold=EVAL["base_threshold"], get_class_metric=True)

    images_path = EVAL["eval_images_path"] if not EVAL["test_mode"] else EVAL["test_images_path"]
    try:
        shutil.rmtree(images_path)
    except:
        pass
    os.mkdir(images_path)

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
        if DATASET["resize"]:
            outputs = torch.nn.functional.interpolate(outputs, size=DATASET["orig_size"], mode='bilinear', align_corners=True)

        outputs = outputs.detach().cpu()
        if not EVAL["test_mode"]:
            meter.update("val", targets, outputs)

        # dump predictions as images
        outputs = (outputs > EVAL["base_threshold"]).int() # thresholding
        outputs = torch2np(outputs)
        pic = image_dataset.label_encoder.class2color(outputs, clean_up_clusters=EVAL["drop_clusters"],
                                                      mode="catId" if DATASET["train_on_cats"] else "trainId")
        if EVAL["images_morphing"]:
            # Add here image+mask morphing
            orig_image = open_img(image_id[0])
            alpha = 0.5
            if (TARGET == "kitti") and (orig_image.shape[:2] != pic.shape[:2]):
                orig_image = cv2.resize(orig_image, (DATASET["orig_size"][1], DATASET["orig_size"][0]), cv2.INTER_LANCZOS4)
            pic = cv2.addWeighted(orig_image, (1 - alpha), pic, alpha, 0)

        pred_name = "_".join(re.split("\.|_", image_id[0].split("/")[-1])[:-1]) + "_predicted_mask.png"
        cv2.imwrite(os.path.join(images_path, pred_name), pic)

    torch.cuda.empty_cache()
    if not EVAL["test_mode"]:
        dices, iou = meter.get_metrics("val")
        print("***** Prediction done in {} sec.; IoU: {}, Dice: {} ***** \n(total elapsed time: {} sec.) ".\
                format(int(time.time()-start), iou, dices[0]["dice_all"], int(time.time()-global_start)))
        if TARGET == "cityscapes" and len(dices[0]) > 1:
            labels_df = image_dataset.label_encoder.cityscapes_labels_df
            if DATASET["train_on_cats"]:
                cat, name = "catId", "category"
            else:
                cat, name = "trainId", "name"
            print("***** Class metrics: *****")
            for k, v in dices[0].items():
                if k != "dice_all":
                    print(labels_df[labels_df[cat] == int(k)][name].iloc[0], " : ", v)
    else:
        print("***** Prediction on test set done in {} sec. ***** \n(total elapsed time: {} sec.) ".\
                format(int(time.time()-start), int(time.time()-global_start)))
