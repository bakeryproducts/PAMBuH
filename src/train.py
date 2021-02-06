import os
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from fastprogress.fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt

import shallow as sh
import utils
from logger import logger
from config import cfg, cfg_init
from data import build_datasets, build_dataloaders
from model import build_model


def upscale(tensor, size): return torch.nn.functional.interpolate(tensor, size=size)

def denorm(images, mean=(0.46454108, 0.43718538, 0.39618185), std=(0.23577851, 0.23005974, 0.23109385)):
    mean=torch.tensor(mean).view((1,3,1,1)).cuda()
    std=torch.tensor(std).view((1,3,1,1)).cuda()
    images = images * std + mean
    return images


class CudaCB(sh.callbacks.Callback):
    def before_batch(self):
        xb,yb = self.batch
        if self.model.training: yb = yb.cuda()
        self.learner.batch = xb.cuda(), yb
            
    def before_fit(self): self.model.cuda()

class TBMetricCB(TrackResultsCB):
    def __init__(self, writer, logger=None, metrics={'totals':['total_loss']}): sh.utils.store_attr(self, locals())
        
    @sh.utils.on_train
    def after_epoch(self):
        n = sum(self.ns)
        self.total_loss = sum(self.losses).item()/n
        for category, metrics in self.metrics.items():
            for metric in metrics:
                self.log_debug(f"{category + '/' + metric, getattr(self, metric), self.n_epoch}")
                self.writer.add_scalar(category + '/' + metric, getattr(self, metric), self.n_epoch)
        self.writer.flush()
        
class DummyCB(sh.callbacks.Callback):
    def __init__(self): sh.utils.store_attr(self, locals())
    
class TBPredictionsCB(sh.callbacks.Callback):
    def __init__(self, writer, logger=None, step=1): sh.utils.store_attr(self, locals())
        
    @sh.utils.on_train
    @sh.utils.on_epoch_step
    def after_epoch(self):
        #self.log_debug('tb predictions')
        count, H, W = 5, 256, 256
        images, gt = self.batch
        preds = self.preds.float()
        images = images[:count].float()

        num_channels = 1

        # YB from batch
        #gt = self.loss_func.buffer[:count,:num_channels,...].float()
        #gt = gt.sum(axis=1, keepdim=True).repeat(1,3,1,1)
        #gt = upscale(gt.detach().cpu(), (H,W))

        img_preds = preds[:count, :num_channels,...]
        img_preds = torch.sigmoid(img_preds)
        img_preds = img_preds.max(1, keepdim=True)[0].repeat(1,3,1,1)
        img_preds = upscale(img_preds.detach().cpu(), (H,W))

        mean, std = self.kwargs['cfg'].TRANSFORMERS.MEAN, self.kwargs['cfg'].TRANSFORMERS.STD 
        images = denorm(images, mean, std)
        images = upscale(images.detach().cpu(), (H,W))

        #self.log_debug(f"{images.shape}, {gt.shape}, {img_preds.shape}")
        summary_image = torch.cat([images, gt, img_preds], 0)
        #self.log_debug(f"{summary_image.shape}")
        grid = torchvision.utils.make_grid(summary_image, nrow=count, pad_value=4)
        self.writer.add_image('predictions', grid, self.n_epoch)
        
        self.writer.flush()
        
class ValCB(sh.callbacks.Callback):
    def __init__(self, logger=None): 
        sh.utils.store_attr(self, locals())
        self.eval_model = None
        self.coco = CocoEval()
        self.evals = []

    @sh.utils.on_validation
    def before_epoch(self):
        self.lucky_numbers = np.random.choice(list(range(len(self.dl))), 3)
        #self.log_debug(f'Indexies for val eval plots:{self.lucky_numbers}')
        self.evals = []

    def evaluate(self, x, y, h):
        # all of them are in batch dim
        # images, gt, heatmaps, predictions
        # Do metric
        metric = 42
        SCALE = 2 # cfg downscale
        mean, std = self.kwargs['cfg'].TRANSFORMERS.MEAN, self.kwargs['cfg'].TRANSFORMERS.STD 
        return 42


    @sh.utils.on_master
    def val_step(self):
        with torch.no_grad():
            xb, yb = self.batch
            hb = self.model(xb)
            metric = self.evaluate(xb, yb, hb)
            #self.log_debug(yb)


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
            
        xb,yb = self.batch
        if self.kwargs['cfg'].TRAIN.AMP_LEVEL:
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



def start(cfg, output_folder, n_epochs):
    device = torch.cuda.current_device()
    
    datasets = build_datasets(num_proc=8)
    dls = build_dataloaders(cfg, datasets, drop_last=True, pin=True)
    
    model, opt = build_model(cfg)
    criterion = BceLoss
    
    
    ########## CALLBACKS #########
    train_cb = TrainCB(logger=logger)
    
    if cfg.PARALLEL.IS_MASTER:
        models_dir = output_folder / 'models'
        tb_dir = output_folder / 'tb'
        step = cfg.TRAIN.TB_STEP
        writer = SummaryWriter(log_dir=tb_dir, comment='Demo')
        
        tb_metric_cb = partial(TBMetricCB, writer=writer, metrics={'totals':['total_loss', 'lr']}, )
        tb_predict_cb = partial(TBPredictionsCB, writer=writer, logger=logger, step=step)

        tb_cbs = [tb_metric_cb(), tb_predict_cb()]
        checkpoint_cb = sh.callbacks.CheckpointCB(models_dir, save_step=1.)
        train_timer_cb = sh.callbacks.TimerCB(mode_train=True, logger=logger)
        master_cbs = [train_timer_cb, *tb_cbs, checkpoint_cb]
    
    l0,l1,l2 = 5e-5, 5e-4,5e-5
    scale = 1#.25
    l0,l1,l2 = l0/scale, l1/scale, l2/scale
    lr_cos_sched = sh.schedulers.combine_scheds([
        [.25, sh.schedulers.sched_cos(l0,l1)],
        [.75, sh.schedulers.sched_cos(l1,l2)]])
    lrcb = sh.callbacks.ParamSchedulerCB('before_epoch', 'lr', lr_cos_sched)
    
    cbs = [CudaCB(), train_cb, lrcb]
        

    if cfg.PARALLEL.IS_MASTER:
        cbs.extend(master_cbs)
        epoch_bar = master_bar(range(n_epochs))
        batch_bar = partial(progress_bar, parent=epoch_bar)
    else: epoch_bar, batch_bar = range(n_epochs), lambda x:x

    learner = sh.learner.Learner(model, opt, sh.utils.AttrDict(dls), criterion, 0, cbs, batch_bar, epoch_bar, cfg=cfg)
    learner.fit(n_epochs)

