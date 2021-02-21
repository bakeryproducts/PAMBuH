import os
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import shallow as sh
import utils
from logger import logger
import loss


def upscale(tensor, size): return torch.nn.functional.interpolate(tensor, size=size)

def denorm(images, mean=(0.46454108, 0.43718538, 0.39618185), std=(0.23577851, 0.23005974, 0.23109385)):
    mean = torch.tensor(mean).view((1,3,1,1))
    std = torch.tensor(std).view((1,3,1,1))
    images = images * std + mean
    return images

def get_xb_yb(b):
    if isinstance(b[1], tuple): return b[0], b[1][0]
    else: return b

def get_tag(b):
    if isinstance(b[1], tuple): return b[1][1]
    else: return None

class CudaCB(sh.callbacks.Callback):
    def before_batch(self):
        xb, yb = get_xb_yb(self.batch)
        tag = get_tag(self.batch)

        yb = yb.cuda()
        labels = yb if not tag else (yb, tag)
        self.learner.batch = xb.cuda(), labels

    def before_fit(self): self.model.cuda()

class TrackResultsCB(sh.callbacks.Callback):
    """
        TODO break alliance with TB metric CB
    """
    def before_epoch(self): 
        self.accs,self.losses,self.samples_count = [],[],[]
        
    def after_epoch(self):
        n = sum(self.samples_count)
        print(self.n_epoch, self.model.training, sum(self.losses)/n, sum(self.accs)/n)
        
    def after_batch(self):
        xb, yb = get_xb_yb(self.batch)
        tag = get_tag(self.batch)
        batch_size = xb.shape[0]
        #print(self.preds.shape, yb.shape, xb.shape)
        p = torch.sigmoid(self.preds.cpu().float())
        p = (p>.5).float()
        dice = loss.dice_loss(p, yb.cpu().float())
        #print(n, dice, dice*n)
        self.accs.append(dice * batch_size)
        self.samples_count.append(batch_size)
        self.losses.append(self.loss.detach().item()*batch_size)


class TBMetricCB(TrackResultsCB):
    def __init__(self, writer, train_metrics=None, validation_metrics=None, logger=None):
        ''' train_metrics = {'losses':['train_loss', 'val_loss']}
            val_metrics = {'metrics':['localization_f1']}
        '''
        sh.utils.store_attr(self, locals())

    def parse_metrics(self, metric_collection):
        if metric_collection is None : return
        for category, metrics in metric_collection.items():
            for metric_name in metrics:
                metric_value = getattr(self, metric_name, None)
                if metric_value is not None: 
                    self.log_debug(f"{category + '/' + metric_name, metric_value, self.n_epoch}")
                    self.writer.add_scalar(category + '/' + metric_name, metric_value, self.n_epoch)


    def after_epoch_train(self):
        #self.log_debug('tb metric after train epoch')
        self.train_loss = sum(self.losses) / sum(self.samples_count)
        self.train_dice =  sum(self.accs) / sum(self.samples_count)
        self.parse_metrics(self.train_metrics)
        
    def after_epoch_valid(self):
        #self.log_debug('tb metric after validation')
        self.val_loss = sum(self.losses) / sum(self.samples_count)
        self.valid_dice =  sum(self.accs) / sum(self.samples_count)
        #self.valid_dice2 =  sum(self.learner.extra_accs) / sum(self.learner.extra_samples_count)
        self.parse_metrics(self.validation_metrics)

    def after_epoch(self):
        if self.model.training: self.after_epoch_train()
        else: self.after_epoch_valid()
        self.writer.flush()

        
class TBPredictionsCB(sh.callbacks.Callback):
    def __init__(self, writer, logger=None, step=1):
        sh.utils.store_attr(self, locals())
        self.count, self.wh = 5, (256, 256)

    def process_batch(self):
        xb, yb = get_xb_yb(self.batch)
        preds = self.preds
        num_channels = 1
        mean, std = self.kwargs['cfg'].TRANSFORMERS.MEAN, self.kwargs['cfg'].TRANSFORMERS.STD 

        xb = xb[:self.count].float().detach().cpu()
        xb = denorm(xb, mean, std)
        xb = upscale(xb, self.wh)

        yb = yb[:self.count].repeat(1,3,1,1)
        yb = upscale(yb.detach().cpu(), self.wh)

        preds = torch.sigmoid(preds[:self.count, ...])
        preds = preds.max(1, keepdim=True)[0].repeat(1,3,1,1)
        preds = upscale(preds.float().detach().cpu(), self.wh)

        return xb, yb, preds
    
    def process_write_predictions(self):
        #self.log_debug('tb predictions')
        xb, yb, preds = self.process_batch() # takes last batch that been used
        #self.log_debug(f"{xb.shape}, {yb.shape}, {preds.shape}")
        summary_image = torch.cat([xb,yb,preds])
        #self.log_debug(f"{summary_image.shape}")
        grid = torchvision.utils.make_grid(summary_image, nrow=self.count, pad_value=4)
        label = 'train predictions' if self.model.training else 'val_predictions'
        self.writer.add_image(label, grid, self.n_epoch)
        self.writer.flush()

    def after_epoch(self):
        if not self.model.training or self.n_epoch % self.step ==0:
            self.process_write_predictions()


class TrainCB(sh.callbacks.Callback):
    def __init__(self, logger=None): 
        sh.utils.store_attr(self, locals())
        self.scaler = torch.cuda.amp.GradScaler() 
    
    @sh.utils.on_train
    def before_epoch(self):
        if self.kwargs['cfg'].PARALLEL.DDP: self.dl.sampler.set_epoch(self.n_epoch)
    
    def train_step(self):
        for i in range(len(self.opt.param_groups)):
            self.learner.opt.param_groups[i]['lr'] = self.lr  
            
        xb, yb = self.batch

        if self.kwargs['cfg'].TRAIN.AMP:
            with torch.cuda.amp.autocast():
                self.learner.preds = self.model(xb)
                self.learner.loss = self.loss_func(self.preds, yb)
            self.scaler.scale(self.learner.loss).backward()
            self.scaler.step(self.learner.opt)
            self.scaler.update()
        else:
            self.learner.preds = self.model(xb)
            self.learner.loss = self.loss_func(self.preds, yb)
            self.learner.loss.backward()
            self.learner.opt.step()
            
        self.learner.opt.zero_grad()

class ValCB(sh.callbacks.Callback):
    def __init__(self, dl=None, logger=None):
        self.extra_valid_dl = dl
        sh.utils.store_attr(self, locals())
        self.evals = []
 
    @sh.utils.on_validation
    def before_epoch(self):
        if self.extra_valid_dl is not None:
            self.run_extra_valid()
        self.learner.metrics = []

    def run_extra_valid(self):
        self.learner.extra_accs, self.learner.extra_samples_count = [], []
        for batch in self.extra_valid_dl:
            xb, yb = get_xb_yb(batch)
            tag = get_tag(batch)
            batch_size = xb.shape[0]

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    preds = self.model(xb)

            #print(preds.shape, yb.shape, xb.shape)
            p = torch.sigmoid(preds.cpu().float())
            p = (p>.5).float()
            dice = loss.dice_loss(p, yb.cpu().float())
            #dice = loss.dice_loss(torch.sigmoid(preds.cpu().float()), yb.cpu().float())
            #print(n, dice, dice*n)
            self.learner.extra_accs.append(dice * batch_size)
            self.learner.extra_samples_count.append(batch_size)
            #self.learner.losses.append(self.loss.detach().item()*batch_size)

    def val_step(self):
        xb, yb = self.batch
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                self.learner.preds = self.model(xb)
                self.learner.loss = self.loss_func(self.preds, yb)

