import os
import time
import datetime
import pickle
from pathlib import Path
from functools import partial
from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import ema
import utils
import loss
import shallow as sh
from logger import logger


def upscale(tensor, size): return torch.nn.functional.interpolate(tensor, size=size)

def denorm(images, mean=(0.46454108, 0.43718538, 0.39618185), std=(0.23577851, 0.23005974, 0.23109385)):
    mean = torch.tensor(mean).view((1,3,1,1))
    std = torch.tensor(std).view((1,3,1,1))
    images = images * std + mean
    return images

def get_xb_yb(b):
    #if isinstance(b[1], tuple): return b[0], b[1][0]
    #else: return b
    return b[0], b[1]

def get_pred(m, xb):
    r = m(xb)
    if isinstance(r, tuple): return r
    else: return r, None

def get_tag(b):
    #if isinstance(b[1], tuple): return b[1][1]
    if len(b) == 3: return b[2]
    else: return None

class CudaCB(sh.callbacks.Callback):
    def before_batch(self):
        xb, yb = get_xb_yb(self.batch)
        bb = get_tag(self.batch)

        yb = yb.cuda()
        if bb is None: self.learner.batch = xb.cuda(), yb
        else: self.learner.batch = xb.cuda(), yb, bb.cuda()

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
        with torch.no_grad():
            xb, yb = get_xb_yb(self.batch)
            tag = get_tag(self.batch)
            batch_size = xb.shape[0]
            #print(self.preds.shape, yb.shape, xb.shape)
            p = torch.sigmoid(self.preds)
            p = (p>.5)
            dice = loss.dice_loss(p.float(), yb.float())
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
        self.max_dice = 0

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
        self.valid_dice2 =  sum(self.learner.extra_accs) / sum(self.learner.extra_samples_count)
        self.parse_metrics(self.validation_metrics)

        if self.valid_dice2 > self.max_dice:
            self.max_dice = self.valid_dice2
            if self.max_dice > .91:
                chpt_cb = get_cb_by_instance(self.learner.cbs, CheckpointCB)
                if chpt_cb is not None: chpt_cb.do_saving(f'cmax_{round(self.max_dice, 4)}')
            
    def after_epoch(self):
        if self.model.training: self.after_epoch_train()
        else: self.after_epoch_valid()
        self.writer.flush()

class EarlyStop(sh.callbacks.Callback):
    def __init__(self, num=5, logger=None):
        sh.utils.store_attr(self, locals())
        self.best = 0
        self.cnt = 0

    @sh.utils.on_validation
    def after_epoch(self):
        cb = get_cb_by_instance(self.cbs, TBMetricCB)
        if cb is None: return 
        best = cb.max_dice
        if best > self.best:
            self.best = best
            self.cnt = 0
        else:
            self.cnt += 1
            if self.cnt >= self.num:
                self.log_debug(f'EARLY STOP, {self.best}')
                #torch.distributed.destroy_process_group() # ugly way , should use allreduce
                raise sh.learner.CancelFitException 
        
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
        yb = upscale(yb.detach().cpu().float(), self.wh)

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
        self.cll = []

    @sh.utils.on_train
    def before_epoch(self):
        if self.kwargs['cfg'].PARALLEL.DDP: self.dl.sampler.set_epoch(self.n_epoch)
        for i in range(len(self.opt.param_groups)):
            self.learner.opt.param_groups[i]['lr'] = self.lr  

    @sh.utils.on_train
    def after_epoch(self):
        #print(np.mean(self.cll))
        self.cll = []
    
    def train_step(self):
        xb, yb = get_xb_yb(self.batch)
        tag = get_tag(self.batch) 
        self.learner.preds, aux_cl_pred = get_pred(self.model, xb)
        loss = self.loss_func(self.preds, yb)

        #if tag is not None:
        if False:
            #print(tag.shape, tag.dtype, aux_cl_pred.shape, aux_cl_pred.dtype)
            cl_mask = yb.sum(axis=(1,2,3)) > 0
            #print(cl_mask)
            a,b = aux_cl_pred[:,0][cl_mask], tag.float()[cl_mask]
            cl_loss = torch.nn.functional.binary_cross_entropy_with_logits(a,b)
            self.cll.append(cl_loss.detach().item())
            #print(cl_loss)
            loss = loss * .5 + .5 * cl_loss

        self.learner.loss = loss

        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), .8)
        self.learner.loss.backward()
        self.learner.opt.step()
        self.learner.opt.zero_grad(set_to_none=True)

        if self.kwargs['cfg'].TRAIN.EMA: self.learner.model_ema.update(self.model)


class TrainSSLCB(sh.callbacks.Callback):
    def __init__(self, ssl_dl, logger=None): 
        sh.utils.store_attr(self, locals())
        self.epoch_start_ssl = 10
        #self.ssl_criterion = loss.symmetric_lovasz
        self.ssl_criterion = torch.nn.BCEWithLogitsLoss()
        self.ssl_loss_scale = 1
        self.high_thr = .8
        self.low_thr = .2

    @sh.utils.on_train
    def before_epoch(self):
        if self.kwargs['cfg'].PARALLEL.DDP: self.dl.sampler.set_epoch(self.n_epoch)
        self.ssl_it = iter(self.ssl_dl)

        for i in range(len(self.opt.param_groups)):
            self.learner.opt.param_groups[i]['lr'] = self.lr  

    def ssl_step(self):
        ssl_loss = None
        if self.n_epoch >= self.epoch_start_ssl:
            try:
                xb, yb = get_xb_yb(next(self.ssl_it))
            except StopIteration:
                print('maxit:', self.np_batch)
                return None

            xb = xb.cuda()
            with torch.no_grad():
                ssl_preds, _ = get_pred(self.learner.model_ema.module, yb.cuda())
                ssl_preds = ssl_preds.sigmoid()

                mask = (ssl_preds < self.low_thr) +  (ssl_preds > self.high_thr) # TODO blur? ema?
                ssl_preds_confident = ssl_preds[mask]

            preds, _ = get_pred(self.model, xb)
            preds_gt = preds[mask]
            if self.np_batch > .9 and np.random.random() > .5: 
                # Randomly drop results to TB
                #print(xbs.shape, ybs.shape, preds.shape)
                self.learner.batch = (xb,ssl_preds)
                self.learner.preds = preds

            ssl_loss = self.ssl_criterion(preds_gt, ssl_preds_confident) * self.ssl_loss_scale

        return ssl_loss

    def train_step(self):
        xb, yb = get_xb_yb(self.batch)
        self.learner.preds , _ = get_pred(self.model, xb)
        base_loss = self.loss_func(self.learner.preds, yb)
        ssl_loss = self.ssl_step()

        if ssl_loss is not None:
            self.log_debug(f'BASE: {base_loss.item()}, SSL:{ssl_loss.item()}')
            loss = (base_loss + ssl_loss) / 2
        else: loss = base_loss

        loss.backward()
        self.learner.loss = loss
        self.learner.opt.step()
        self.learner.opt.zero_grad(set_to_none=True)

        if self.kwargs['cfg'].TRAIN.EMA: self.learner.model_ema.update(self.model, self.kwargs['cfg'].TRAIN.EMA)


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
        self.reduction = 'none' if self.kwargs['cfg'].TRAIN.SELECTIVE_BP <1 - 1e-6 else 'mean'

    def run_extra_valid(self):
        self.learner.extra_accs, self.learner.extra_samples_count = [], []
        for batch in self.extra_valid_dl:
            xb, yb = get_xb_yb(batch)
            tag = get_tag(batch)
            batch_size = xb.shape[0]

            with torch.no_grad():
                preds , _ = get_pred(self.model, xb)

            #print(preds.shape, yb.shape, xb.shape)
            p = torch.sigmoid(preds.cpu().float())
            p = (p>.5).float()
            dice = loss.dice_loss(p, yb.cpu().float())
            #dice = loss.dice_loss(torch.sigmoid(preds.cpu().float()), yb.cpu().float())
            #print(n, dice, dice*n)
            self.learner.extra_accs.append(dice * batch_size)
            self.learner.extra_samples_count.append(batch_size)

    def val_step(self):
        xb, yb = get_xb_yb(self.batch)
        with torch.no_grad():
            #with torch.cuda.amp.autocast():
            self.learner.preds, _ = get_pred(self.model, xb)
            self.learner.loss = self.loss_func(self.preds, yb)
                

class ValEMACB(sh.callbacks.Callback):
    def __init__(self, model_ema, logger=None):
        sh.utils.store_attr(self, locals())
        self.evals = []

    def before_fit(self):
        self.learner.model_ema = self.model_ema

    @sh.utils.on_validation
    def before_epoch(self):
        self.run_ema_valid()
        self.learner.metrics = []

    def run_ema_valid(self):
        self.learner.extra_accs, self.learner.extra_samples_count = [], []
        for batch in self.dls['VALID']:
            xb, yb = get_xb_yb(batch)
            tag = get_tag(batch)
            batch_size = xb.shape[0]

            with torch.no_grad():
                preds, _ = get_pred(self.learner.model_ema.module, xb.cuda())

            p = torch.sigmoid(preds.cpu().float())
            p = (p>.5).float()
            dice = loss.dice_loss(p, yb.float())
            self.learner.extra_accs.append(dice * batch_size)
            self.learner.extra_samples_count.append(batch_size)


    def val_step(self):
        xb, yb = get_xb_yb(self.batch)
        with torch.no_grad():
            self.learner.preds , _ = get_pred(self.model, xb)
            self.learner.loss = self.loss_func(self.preds, yb)
                

def get_cb_by_instance(cbs, cls):
    for cb in cbs:
        if isinstance(cb, cls): return cb
    return None

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()

class CheckpointCB(sh.callbacks.Callback):
    def __init__(self, save_path, ema=False, save_step=None):
        sh.utils.store_attr(self, locals())
        self.pct_counter = None if isinstance(self.save_step, int) else self.save_step
        
    def do_saving(self, val=''):
        state_dict = get_state_dict(self.model) if not self.ema else get_state_dict(self.model_ema)
        postfix = 'ema_' if self.ema else ''
        torch.save({
                'epoch': self.n_epoch,
                'loss': self.loss,
                'model_state': state_dict,
                'opt_state':self.opt.state_dict(), 
            }, str(self.save_path / f'e{self.n_epoch}_t{self.total_epochs}_{postfix}{val}.pth'))

    def after_epoch(self):
        save = False
        if self.n_epoch == self.total_epochs - 1: save=True
        elif isinstance(self.save_step, int): save = self.save_step % self.n_epoch == 0 
        else:
            if self.np_epoch > self.pct_counter:
                save = True
                self.pct_counter += self.save_step
        
