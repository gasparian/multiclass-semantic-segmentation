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
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

class Meter:

    '''A meter to keep track of iou and dice scores throughout an epoch'''

    def __init__(self, root_result_dir="", base_threshold=.5, get_class_metric=False):
        self.base_threshold = base_threshold # threshold
        self.get_class_metric = get_class_metric
        # tensorboard logging
        if root_result_dir:
            self.tb_log = SummaryWriter(log_dir=os.path.join(root_result_dir, 'tensorboard'))
        self.reset_dicts()
    
    def reset_dicts(self):
        self.base_dice_scores = {"train":[], "val":[]}
        self.dice_neg_scores = {"train":[], "val":[]}
        self.dice_pos_scores = {"train":[], "val":[]}
        self.iou_scores = {"train":[], "val":[]}

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
            probability = (probability > self.base_threshold).float()
            truth = (truth > 0.5).float()

            p = probability.view(batch_size, -1)
            t = truth.view(batch_size, -1)
            assert(p.shape == t.shape)

            intersection = (p*t).sum(-1)
            union = (p+t).sum(-1)

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            neg = (p_sum == 0).float()
            dice_pos = (2 * intersection) / (union + 1e-7)
            iou_pos = intersection / (union + 1e-7)

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

            dice = {"dice_all": dice}

            if self.get_class_metric:
                num_classes = probability.shape[1]
                for c in range(num_classes):
                    iflat = probability[:, c,...].view(batch_size, -1)
                    tflat = truth[:, c,...].view(batch_size, -1)
                    intersection = (iflat * tflat).sum()
                    dice[str(c)] = ((2. * intersection) / (iflat.sum() + tflat.sum() + 1e-7)).item()

        return iou, dice, neg, dice_pos, num_neg, num_pos

    def update(self, phase, targets, outputs):
        """updates metrics lists every iteration"""
        iou, dice, dice_neg, dice_pos, _, _ = self.metric(outputs, targets)
        self.base_dice_scores[phase].append(dice)
        self.dice_pos_scores[phase].append(dice_pos)
        self.dice_neg_scores[phase].append(dice_neg)
        self.iou_scores[phase].append(iou)

    def get_metrics(self, phase):
        """averages computed metrics over the epoch"""
        dice = {}
        l = len(self.base_dice_scores[phase])
        for i, d in enumerate(self.base_dice_scores[phase]):
            for k in d:
                if k not in dice:
                    dice[k] = 0
                dice[k] += d[k] / l
            
        dice_neg = np.mean(self.dice_neg_scores[phase])
        dice_pos = np.mean(self.dice_pos_scores[phase])
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores[phase])
        return dices, iou

    def epoch_log(self, phase, epoch_loss, itr):
        '''logging the metrics at the end of an epoch'''
        dices, iou = self.get_metrics(phase)
        dice, dice_neg, dice_pos = dices
        message = "Phase: %s | Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" \
            % (phase, epoch_loss, iou, dice["dice_all"], dice_neg, dice_pos)
        logging.info(message)

        self.tb_log.add_scalar(f'{phase}_dice', dice["dice_all"], itr)
        self.tb_log.add_scalar(f'{phase}_dice_neg', dice_neg, itr)
        self.tb_log.add_scalar(f'{phase}_dice_pos', dice_pos, itr)
        self.tb_log.add_scalar(f'{phase}_iou', iou, itr)
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
                 smooth=.0, class_weights=[], threshold=0., activate=False):

        self.bce_weight = bce_weight
        self.eps = eps
        self.smooth = smooth
        self.threshold = threshold # 0 or apply sigmoid and threshold > .5 instead
        self.activate = activate
        self.class_weights = class_weights
        self.nll = torch.nn.BCEWithLogitsLoss(weight=weight)

    def __call__(self, logits, true):
        loss = self.bce_weight * self.nll(logits, true)
        if self.bce_weight < 1.:
            dice_loss = 0.
            batch_size, num_classes = logits.shape[:2]
            if self.activate:
                logits = torch.sigmoid(logits)
            logits = (logits > self.threshold).float()
            for c in range(num_classes):
                iflat = logits[:, c,...].view(batch_size, -1)
                tflat = true[:, c,...].view(batch_size, -1)
                intersection = (iflat * tflat).sum()
                
                w = self.class_weights[c]
                dice_loss += w * ((2. * intersection + self.smooth) /
                                 (iflat.sum() + tflat.sum() + self.smooth + self.eps))
            loss -= (1 - self.bce_weight) * torch.log(dice_loss)

        return loss

def fix_multigpu_chkpt_names(state_dict, drop=False):
    """ fix the DataParallel caused problem with keys names """
    new_state_dict = {}
    for k in state_dict:
        if drop:
            new_k = re.sub("module.", "", k)
        else:
            new_k = "module." + k
        new_state_dict[new_k] = copy.deepcopy(state_dict[k])
    return new_state_dict

class Trainer(object):
    
    '''Basic functionality for models fitting'''
    
    __params = ('num_workers', 'class_weights', 'accumulation_batches',
                'lr', 'weights_decay', 'base_threshold', 'scheduler_patience', 'activate',
                'freeze_n_iters', 'bce_loss_weight', 'key_metric')
    
    def __init__(self, model=None, image_dataset=None, optimizer=None, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        self.possible_phases = ["train", "val", "test"]

        # Initialize logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s : %(levelname)s : %(message)s'
        )

        if not isinstance(self.devices_ids, list):
            self.devices_ids = list(self.devices_ids)

        # Run two model's instances on multiple GPUs ##############################################
        # seems like multi-GPU mode works fine only for torch==1.1.0 
        # in other cases - try `DistributedDataParallel` https://github.com/pytorch/examples/blob/master/imagenet/main.py

        if torch.cuda.is_available():
            main_device = "cuda:%i" % self.devices_ids[0]
        else:
            main_device = "cpu"
        self.device = torch.device(main_device)

        self.net = model
        self.multi_gpu_flag = (torch.cuda.device_count() > 1) * (len(self.devices_ids) > 1)
        if self.multi_gpu_flag:
            self.net = nn.DataParallel(self.net, device_ids=self.devices_ids, output_device=self.devices_ids[0])
        self.net.to(self.device)

        ############################################################################################
        
        self.best_metric = float("inf")
        self.phases = ["train", "val"]
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.freeze_flag = True
        self.start_epoch = 0
        cudnn.benchmark = True

        self.image_dataset = image_dataset
        self.criterion = BCEDiceLoss(bce_weight=self.bce_loss_weight, class_weights=self.class_weights, 
                                     threshold=self.base_threshold, activate=self.activate)
        self.optimizer = optimizer(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=self.scheduler_patience, verbose=True)
        
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

        self.meter = Meter(self.model_path, self.base_threshold)

        if self.load_checkpoint:
            self.load_model(ckpt_name=self.load_checkpoint)

        self.accumulation_steps = self.batch_size * self.accumulation_batches

        # number of workers affect the GPU performance if the preprocessing too intensive (resizes \ augs)
        self.num_workers = max(2, self.batch_size // 2)
        # self.num_workers = self.batch_size

        logging.info(f"Trainer initialized on {len(self.devices_ids)} devices!")

    def load_model(self, ckpt_name="best_model.pth"):
        """Loads full model state and basic training params"""
        path = "/".join(ckpt_name.split("/")[:-1])
        chkpt = torch.load(ckpt_name)
        self.start_epoch = chkpt['epoch']
        self.best_metric = chkpt['best_metric']

        # fix the DataParallel caused problem with keys names
        if self.multi_gpu_flag:
            new_state_dict = fix_multigpu_chkpt_names(chkpt['state_dict'], drop=False)
            self.net.load_state_dict(new_state_dict)
        else:
            try:
                self.net.load_state_dict(chkpt['state_dict'])
            except:
                new_state_dict = fix_multigpu_chkpt_names(chkpt['state_dict'], drop=True)
                self.net.load_state_dict(new_state_dict)

        if self.load_optimizer_state:
            self.optimizer.load_state_dict(chkpt['optimizer'])
        logging.info("******** State loaded ********")

        training_meta = pickle.load(open(f"{path}/training_meta.pickle.dat", "rb"))
        for k, v in training_meta.items():
            if k in self.__class__.__params:
                setattr(self, k, v)
        logging.info("******** Training params loaded ********")

    def forward(self, images, targets):
        """allocate data and runs forward pass through the network"""
        # send all variables to selected device
        images = images.to(self.device)
        masks = targets.to(self.device)
        # compute loss
        outputs = self.net(images)
        orig_size = self.image_dataset.orig_size
        if outputs.size()[-2:] != orig_size:
            # resize predictions back to the original size
            outputs = nn.functional.interpolate(outputs, size=orig_size, mode='bilinear', align_corners=True)
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
                param.data = param.data.add(-1.*self.weights_decay * param_group['lr'], param.data)

    def iterate(self, epoch, phase, data_set):
        """main method for traning: creates metric aggregator, dataloaders and updates the model params"""

        if phase not in self.possible_phases:
            raise ValueError('Phase type must be on of: {}'.format(self.possible_phases))

        self.meter.reset_dicts()
        start = time.strftime("%H:%M:%S")
        logging.info(f"Starting epoch: {epoch} | phase: {phase} | time: {start}")
        self.net.train(phase == "train")
    
        self.image_dataset.set_phase(phase, data_set)
        dataloader = DataLoader(
            self.image_dataset,
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
                
            images, targets, __ = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    if self.weights_decay > 0:
                        self.weights_decay()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()
            if self.activate:
                outputs = torch.sigmoid(outputs)
            outputs = outputs.detach().cpu()
            self.meter.update(phase, targets, outputs)
            running_loss_tick = (running_loss * self.accumulation_steps) / (itr + 1)

            self.meter.tb_log.add_scalar(f'{phase}_loss', running_loss_tick, itr+total_batches*epoch)
            tk0.set_postfix(loss=(running_loss_tick))

        last_itr = itr+total_batches*epoch
            
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = self.meter.epoch_log(phase, epoch_loss, last_itr)

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice["dice_all"])
        self.iou_scores[phase].append(iou)

        torch.cuda.empty_cache()
        return epoch_loss, dice, iou, last_itr
    
    def make_new_dir(self):
        """makes new directory instead of existing one"""
        try:
            shutil.rmtree(self.model_path)
        except:
            pass
        os.mkdir(self.model_path)
        
    def dump_meta(self):
        """dumpы all metrics and training meta-data"""
        for dict_name in ["losses", "dice_scores", "iou_scores"]:
            pickle.dump(self.__dict__[dict_name], open(f"{self.model_path}/{dict_name}.pickle.dat", "wb"))

        # dump class variables for further traning
        training_meta = {}
        for k in self.__class__.__params:
            training_meta[k] = getattr(self, k)
        pickle.dump(training_meta, open(f"{self.model_path}/training_meta.pickle.dat", "wb"))

    def get_current_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            return float(param_group['lr'])

    def start(self, train_set, val_set):
        """Runs training loop and saves intermediate state"""
            
        for epoch in range(self.start_epoch, self.num_epochs+self.start_epoch):
            _, __, ___, last_itr = self.iterate(epoch, "train", train_set)
            self.meter.tb_log.add_scalar(f'learning_rate', self.get_current_lr(), last_itr)

            state = {
                "epoch": epoch,
                "best_metric": self.best_metric,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            
            with torch.no_grad():
                val_loss, val_dice, val_iou, __ = self.iterate(epoch, "val", val_set)
                key_metrics = {"loss": val_loss, "dice": -1.*val_dice, "iou": -1.*val_iou} # -1* for the scheduler
                self.scheduler.step(val_loss)

            is_last_epoch = epoch == (self.num_epochs - 1)
            if key_metrics[self.key_metric] < self.best_metric or is_last_epoch:
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

                ckpt_name = "best_model"
                if is_last_epoch:
                    ckpt_name = "last_model"
                torch.save(state, f"{self.model_path}/{ckpt_name}.pth")
                
        # save meta-data into the workdir
        self.dump_meta()
