import os
import gc
import re
import time
import copy
import shutil
import pickle
import logging

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class Meter:

    '''A meter to keep track of iou and dice scores throughout an epoch'''

    def __init__(self, phase, epoch, base_threshold=.5):
        self.base_threshold = base_threshold # threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def predict(self, X):
        '''X is sigmoid output of the model'''
        X_p = np.copy(X)
        preds = (X_p > self.base_threshold).astype('uint8')
        return preds

    def metric(self, probability, truth):
        '''Calculates dice of positive and negative images seperately'''
        '''probability and truth must be torch tensors'''
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert(probability.shape == truth.shape)

            p = (probability > self.base_threshold).float()
            t = (truth > 0.5).float()
            intersection = (p*t).sum(-1)
            union = (p+t).sum(-1)

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            neg = (p_sum == 0).float()
            dice_pos = (2 * intersection) / union
            iou_pos = intersection / union

            neg = neg[neg_index]
            dice_pos = dice_pos[pos_index]
            iou_pos = iou_pos[pos_index]

            dice = torch.cat([dice_pos, neg])
            iou = torch.cat([iou_pos, neg])

            neg = np.nan_to_num(neg.mean().item(), 0)
            dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)

            dice = dice.mean().item()
            iou = iou.mean().item()

            num_neg = len(neg_index)
            num_pos = len(pos_index)

        return iou, dice, neg, dice_pos, num_neg, num_pos

    def update(self, targets, outputs):
        """updates metrics lists every iteration"""
        iou, dice, dice_neg, dice_pos, _, _ = self.metric(outputs, targets)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        self.iou_scores.append(iou)

    def get_metrics(self):
        """averages computed metrics over the epoch"""
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

    def epoch_log(self, epoch_loss):
        '''logging the metrics at the end of an epoch'''
        dices, iou = self.get_metrics()
        dice, dice_neg, dice_pos = dices
        message = "Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos)
        logging.info(message)
        return dice, iou

class BCEDiceLoss:

    """
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: loss.
    """

    def __init__(self, bce_weight=1., weight=None, eps=1e-7, 
                 smooth=1., class_weights=[]):

        self.bce_weight = bce_weight
        self.eps = eps
        self.smooth = smooth
        self.class_weights = class_weights
        self.nll = torch.nn.BCEWithLogitsLoss(weight=weight)

    def __call__(self, logits, true):
        loss = self.bce_weight * self.nll(logits, true)
        if self.bce_weight < 1.:
            dice_loss = 0.
            batch_size, num_classes = logits.shape[:2] 
            logits = (logits > 0).float() # ...or apply sigmoid and threshold > .5 instead
            # TO DO: vectorize the code below
            for c in range(num_classes):
                iflat = logits[:, c,...].view(batch_size, -1)
                tflat = true[:, c,...].view(batch_size, -1)
                intersection = (iflat * tflat).sum()
                
                w = self.class_weights[c]
                dice_loss += w * ((2. * intersection + self.smooth) /
                                 (iflat.sum() + tflat.sum() + self.smooth + self.eps))
            loss -= (1 - self.bce_weight) * torch.log(dice_loss)

        return loss

class Trainer(object):
    
    '''Basic functionality for models fitting'''
    
    __params = ('num_workers', 'model_path', 'batch_size', 'class_weights', 'accumulation_batches',
                'lr', 'wd', 'base_threshold', 'scheduler_patience', 'activate',
                'num_epochs', 'resize', 'freeze_n_iters', 'bce_loss_weight', 'key_metric')
    
    def __init__(self, model=None, model_path='', dataset_processor=None, resize=None,
                 reset=True, batch_size=4, freeze_n_iters=20, weights_decay=5e-5,
                 lr=5e-4, num_epochs=20, bce_loss_weight=.9, devices_ids=[0],
                 accumulation_batches=4, key_metric="dice", optimizer=None, 
                 base_threshold=.0, scheduler_patience=3, activate=False,
                 class_weights=[1/4 for i in range(4)]):

        # Initialize logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s : %(levelname)s : %(message)s'
        )

        self.devices_ids = devices_ids
        if not isinstance(self.devices_ids, list):
            self.devices_ids = list(self.devices_ids)

        # Run two model's instances on multiple GPUs ##############################################
        # seems like multi-GPU mode works fine only for torch==1.1.0 
        # in other cases - try `DistributedDataParallel` https://github.com/pytorch/examples/blob/master/imagenet/main.py

        if torch.cuda.is_available():
            main_device = "cuda:%i" % devices_ids[0]
        else:
            main_device = "cpu"
        self.device = torch.device(main_device)

        self.net = model
        self.multi_gpu_flag = (torch.cuda.device_count() > 1) * (len(self.devices_ids) > 1)
        if self.multi_gpu_flag:
            self.net = nn.DataParallel(self.net, device_ids=devices_ids, output_device=devices_ids[0])
        self.net.to(self.device)

        ############################################################################################
        
        self.best_metric = float("inf")
        self.phases = ["train", "val"]
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.freeze_flag = True
        self.key_metric = key_metric
        self.activate = activate
        self.freeze_n_iters = freeze_n_iters
        self.base_threshold = base_threshold
        self.bce_loss_weight = bce_loss_weight
        self.class_weights = class_weights
        self.accumulation_batches = accumulation_batches
        self.batch_size = batch_size
        self.num_workers = self.batch_size
        self.scheduler_patience = scheduler_patience
        self.lr = lr
        self.wd = weights_decay
        self.resize = resize # tuple of new sized, for ex.: (800, 128)
        self.num_epochs = num_epochs
        self.model_path = model_path
        cudnn.benchmark = True
        
        self.accumulation_steps = self.batch_size * self.accumulation_batches
        self.reset = reset
        self.dataset_processor = dataset_processor
        self.criterion = BCEDiceLoss(bce_weight=self.bce_loss_weight, class_weights=self.class_weights)
        self.optimizer = optimizer(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=self.scheduler_patience, verbose=True)
        
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

        logging.info(f"Trainer initialized on {len(self.devices_ids)} devices!")

    def forward(self, images, targets):
        """allocate data and runs forward pass through the network"""
        # send all variables to selected device
        images = images.to(self.device)
        masks = targets.to(self.device)

        # compute loss
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs
    
    def dfs_freeze(self, model):
        """freezes weights of the input layer"""
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False if self.freeze_flag else True
            self.dfs_freeze(child)
            
    # freezes UNet encoder
    # doesn't work properely in Dataparallel mode
    # since it's wrapps our model class
    def freeze_encoder(self):
        """freezes encoder module in order to train the other part of the network"""
        self.dfs_freeze(self.net.conv1)
        self.dfs_freeze(self.net.conv2)
        self.dfs_freeze(self.net.conv3)
        self.dfs_freeze(self.net.conv4)
        self.dfs_freeze(self.net.conv5)

    def weights_decay(self):
        """adjust learning rate and weights decay"""
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.data = param.data.add(-self.wd * param_group['lr'], param.data)

    def iterate(self, epoch, phase, data_set):
        """main method for traning: creates metric aggregator, dataloaders and updates the model params"""
        meter = Meter(phase, epoch, self.base_threshold)
        start = time.strftime("%H:%M:%S")
        logging.info(f"Starting epoch: {epoch} | phase: {phase} | time: {start}")
        self.net.train(phase == "train")
        
        image_dataset = self.dataset_processor(data_set, phase, self.resize)
        dataloader = DataLoader(
            image_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,   
        )
        
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        tk0 = tqdm(dataloader, total=total_batches)
        if self.freeze_n_iters:
            self.freeze_encoder()
            
        for itr, batch in enumerate(tk0):
            
            if itr == (self.freeze_n_iters - 1):
                self.freeze_flag = False
                self.freeze_encoder()
                
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    if self.wd > 0:
                        self.weights_decay()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()
            if self.activate:
                outputs = torch.sigmoid(outputs)
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
            running_loss_tick = (running_loss * self.accumulation_steps) / (itr + 1)
            tk0.set_postfix(loss=(running_loss_tick))
            
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = meter.epoch_log(epoch_loss)

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)

        torch.cuda.empty_cache()
        return epoch_loss, dice, iou
    
    def make_new_dir(self):
        """makes new directory, if needed"""
        try:
            shutil.rmtree(self.model_path)
        except:
            pass
        os.mkdir(self.model_path)
        
    def dump_meta(self):
        """dumpы all metrics and training meta-data"""
        for dict_name in ["losses", "dice_scores", "iou_scores"]:
            pickle.dump(self.__dict__[dict_name], open(f"{self.model_path}/{dict_name}.pickle.dat", "wb"))

        training_meta = {}
        for k in self.__class__.__params:
            training_meta[k] = getattr(self, k)
        pickle.dump(training_meta, open(f"{self.model_path}/training_meta.pickle.dat", "wb"))

    def start(self, train_set, val_set):
        """Runs training loop and saves intermediate state"""
        if self.reset:
            self.make_new_dir()
            
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train", train_set)
            state = {
                "epoch": epoch,
                "best_metric": self.best_metric,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            
            with torch.no_grad():
                val_loss, val_dice, val_iou = self.iterate(epoch, "val", val_set)
                key_metrics = {"loss": val_loss, "dice": -1.*val_dice, "iou": -1.*val_iou}
                self.scheduler.step(val_loss)
                
            if key_metrics[self.key_metric] < self.best_metric:
                logging.info("******** Saving state ********")

                if self.multi_gpu_flag:
                    # fix parameters names in a state dict. (caused by nn.DataParallel)
                    new_state_dict = {}
                    for k in state["state_dict"]:
                        if k.startswith("module."):
                            new_k = re.sub("module.", "", k)
                            new_state_dict[new_k] = copy.deepcopy(state["state_dict"][k])
                    del state["state_dict"]
                    gc.collect()
                    state["state_dict"] = new_state_dict

                state["val_IoU"] = val_iou
                state["val_dice"] = val_dice
                state["val_loss"] = val_loss
                state["best_metric"] = self.best_metric = key_metrics[self.key_metric]
                torch.save(state, f"{self.model_path}/model.pth")
                
        # save meta-data into the workdir
        self.dump_meta()
