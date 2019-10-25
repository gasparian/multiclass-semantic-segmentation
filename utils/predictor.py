# NOT IMPLEMENTED

train_masks_area = pd.read_csv("./data/train_masks_area.csv", dtype=object, index_col=None)
#train_df = pd.read_csv("./data/train.csv")
train_imgs, val_imgs = \
    train_test_split(train_masks_area["ImageId"].values, test_size=.2, stratify=train_masks_area["area_hash"], 
                         shuffle=True, random_state=42)


model = FPN(encoder_name='resnext50',
            pretrained=False,
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            classes=4,
            dropout=0.2,
            activation='sigmoid',
            final_upsampling=4,
            decoder_merge_policy='add')

device = torch.device("cuda")
model.to(device)
model.eval()
state = torch.load("./FPN_tests/model.pth", map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

TTAModel = TTAWrapper(model, 
                      merge_mode="gmean", 
                      activate=False, 
                      temperature=0.5)

image_dataset = SteelDataset(val_imgs, "val")
dataloader = DataLoader(
    image_dataset,
    batch_size=2,
    num_workers=4,
    pin_memory=True,
    shuffle=True,   
)

meter = Meter("val", 0, 0.)

for batch in tqdm(dataloader):
    images, is_mask_target, targets = batch

    images = images.to(device)
    outputs = TTAModel(images)
    # outputs, __ = model(images)
    # outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()

    meter.update(targets, outputs)

torch.cuda.empty_cache()

dices, iou = meter.get_metrics()
print(dices, iou)