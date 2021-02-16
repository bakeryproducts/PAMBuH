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


def upscale(tensor, size): return torch.nn.functional.interpolate(tensor, size=size)

def denorm(images, mean=(0.46454108, 0.43718538, 0.39618185), std=(0.23577851, 0.23005974, 0.23109385)):
    mean = torch.tensor(mean).view((1,3,1,1))
    std = torch.tensor(std).view((1,3,1,1))
    images = images * std + mean
    return images

class CudaCB(sh.callbacks.Callback):
    def before_batch(self):
        xb, yb = self.batch
        if self.model.training: yb = yb.cuda()
        self.learner.batch = xb.cuda(), yb

    def before_fit(self): self.model.cuda()

class TrackResultsCB(sh.callbacks.Callback):
    def before_epoch(self): 
        self.accs,self.losses,self.samples_count = [],[],[]
        
    def after_epoch(self):
        n = sum(self.samples_count)
        print(self.n_epoch, self.model.training, sum(self.losses)/n, sum(self.accs)/n)
        
    def after_batch(self):
        xb, yb = self.batch
        n = xb.shape[0]
        #print(self.preds.shape, yb.shape, xb.shape)
        dice = dice_loss(torch.sigmoid(self.preds.cpu().float()), yb.cpu().float())
        #print(n, dice, dice*n)
        self.accs.append(dice*n)
        self.samples_count.append(n)
        if self.model.training:
            self.losses.append(self.loss.detach().item()*n)


def dice_loss(inp, target, eps=1e-6):
    with torch.no_grad():
        a = inp.contiguous().view(-1)
        b = target.contiguous().view(-1)
        intersection = (a * b).sum()
        dice = ((2. * intersection + eps) / (a.sum() + b.sum() + eps)) 
        return dice.item()

def focal_loss(outputs: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
    """
    Compute binary focal loss between target and output logits.

    Args:
        reduction (string, optional):
            none | mean | sum | batchwise_mean
            none: no reduction will be applied,
            mean: the sum of the output will be divided by the number of elements in the output,
            sum: the output will be summed.
    Returns:
        computed loss

    Source: https://github.com/BloodAxe/pytorch-toolbelt, Catalyst
    """
    targets = targets.type(outputs.type())
    logpt = -torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt
    #if alpha is not None: loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))

    if reduction == "mean": loss = loss.mean()
    if reduction == "sum": loss = loss.sum()
    if reduction == "batchwise_mean": loss = loss.sum(0)

    return loss


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
        self.valid_dice =  sum(self.accs) / sum(self.samples_count)
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
        xb, yb = self.batch
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
    def __init__(self, logger=None, use_cuda=True): 
        sh.utils.store_attr(self, locals())
        self.scaler = torch.cuda.amp.GradScaler() 
    
    @sh.utils.on_train
    def before_epoch(self):
        if self.kwargs['cfg'].PARALLEL.DDP: self.dl.sampler.set_epoch(self.n_epoch)
    
    def train_step(self):
        for i in range(len(self.opt.param_groups)):
            self.learner.opt.param_groups[i]['lr'] = self.lr  
            
        xb,yb = self.batch
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
    def __init__(self, logger=None):
        sh.utils.store_attr(self, locals())
        self.evals = []
 
    @sh.utils.on_validation
    def before_epoch(self):
        self.lucky_numbers = np.random.choice(list(range(len(self.dl))), 3)
        #self.log_debug(f'Indexies for val eval plots:{self.lucky_numbers}')
        self.evals = []
        self.learner.metrics = []

    def evaluate(self, xb, yb, preds):
        # all of them are in batch dim
        if self.n_batch in self.lucky_numbers:
            pass
        return metric
                
    @sh.utils.on_master
    def val_step(self):
        with torch.no_grad():
            #self.log_debug('VALSTEP')
            xb, yb = self.batch
            self.learner.preds = self.model(xb)
            #metric = self.evaluate(xb, yb, preds)





