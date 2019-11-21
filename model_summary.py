import re
import argparse

from torchsummary import summary
from utils import UnetResNet, FPN
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=False, default=8)
    parser.add_argument('--unet_res_blocks', type=int, required=False, default=0)
    parser.add_argument('--input_size', type=str, required=False, default="3,512,1024")
    args = parser.parse_args()
    globals().update(args.__dict__)

    model_type = model_type.lower()
    input_size = tuple(int(d) for d in re.split("\D", input_size))

    if model_type == "unet":
        model = UnetResNet(encoder_name=backbone, 
                           num_classes=num_classes, 
                           input_channels=3, 
                           num_filters=32, 
                           Dropout=0.3, 
                           res_blocks_dec=bool(unet_res_blocks))

    elif model_type == "fpn":
        model = FPN(encoder_name=backbone,
                    decoder_pyramid_channels=256,
                    decoder_segmentation_channels=128,
                    classes=num_classes,
                    dropout=0.3,
                    activation='sigmoid',
                    final_upsampling=4,
                    decoder_merge_policy='add')
    else:
        raise ValueError('Model type is not correct: `{}`.'.format(model_type))

    device = torch.device("cpu")
    model = model.to(device)

    summary(model, input_size=input_size, device="cpu")