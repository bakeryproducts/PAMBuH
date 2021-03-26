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
        with torch.no_grad():
            xb, yb = get_xb_yb(self.batch)
            tag = get_tag(self.batch)
            batch_size = xb.shape[0]
            #print(self.preds.shape, yb.shape, xb.shape)
            p = torch.sigmoid(self.preds)
            p = (p>.4)
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


class SelectiveTrainCB(sh.callbacks.Callback):
    def __init__(self, logger=None): 
        sh.utils.store_attr(self, locals())
        self.scaler = torch.cuda.amp.GradScaler() 
        self._timestamp = '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())
        ('losses' / Path(self._timestamp)).mkdir()
        self.counter = defaultdict(int)

    def before_fit(self): self.sbp_pct = self.kwargs['cfg'].TRAIN.SELECTIVE_BP
    
    @sh.utils.on_train
    def before_epoch(self):
        if self.kwargs['cfg'].PARALLEL.DDP: self.dl.sampler.set_epoch(self.n_epoch)
        self.learner.unreduced_loss = []
    
    def train_step(self):
        for i in range(len(self.opt.param_groups)): self.learner.opt.param_groups[i]['lr'] = self.lr  
            
        xb, yb = self.batch
        with torch.cuda.amp.autocast():
            self.learner.preds = self.model(xb)
            u_loss = self.loss_func(self.preds, yb, reduction='none')
            self.learner.loss = u_loss.mean() 

        if self.np_batch <= self.sbp_pct:
            self.scaler.scale(self.learner.loss).backward()
            self.scaler.step(self.learner.opt)
            self.scaler.update()
            
        self.learner.opt.zero_grad()
        self.learner.unreduced_loss.extend(u_loss.detach().cpu())

    @sh.utils.on_train
    def after_epoch(self):
        l = torch.tensor(self.learner.unreduced_loss)
        self.learner.dls['TRAIN'].sampler.set_weights(self.losses_to_weights(l))


        if self.kwargs['cfg'].PARALLEL.IS_MASTER:
            ws = np.zeros_like(self.learner.dls['TRAIN'].sampler.sampler.weights)
            c = 0 
            for w, idx in zip(l, self.learner.dls['TRAIN'].sampler.dataset.sampler_list):
                ws[idx] = w
                if c < l.shape[0] * self.kwargs['cfg'].TRAIN.SELECTIVE_BP:
                    self.counter[idx]+=1
                c+=1

            name = f'losses/{self._timestamp}/ll_' + f'{self.n_epoch}'.zfill(5) + '.npy'
            with open(name, 'wb') as f:
                np.save(f, ws)
            name = f'losses/{self._timestamp}/ll_' + f'{self.n_epoch}'.zfill(5) + '.pkl'
            with open(name, 'wb') as f:
                pickle.dump(self.counter, f)

    def losses_to_weights(self, l):
        #return  (1+l)**2
        return  np.percentile(l.numpy(), l)

class TrainCB(sh.callbacks.Callback):
    def __init__(self, logger=None): 
        sh.utils.store_attr(self, locals())
        self.scaler = torch.cuda.amp.GradScaler() 

    def before_fit(self): 
        assert self.kwargs['cfg'].TRAIN.AMP

    @sh.utils.on_train
    def before_epoch(self):
        #print('SCALE:', self.scaler.get_scale())

        if self.kwargs['cfg'].PARALLEL.DDP: self.dl.sampler.set_epoch(self.n_epoch)
        for i in range(len(self.opt.param_groups)):
            self.learner.opt.param_groups[i]['lr'] = self.lr  
    
    def train_step(self):
        xb, yb = self.batch
        with torch.cuda.amp.autocast():
            self.learner.preds = self.model(xb)
            self.learner.loss = self.loss_func(self.preds, yb)
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

        #self.learner.loss.backward()
        self.scaler.scale(self.learner.loss).backward()
        self.scaler.step(self.learner.opt)
        self.scaler.update()
        #print('SCALE:', self.scaler.get_scale())
        #self.learner.opt.step()
        self.learner.opt.zero_grad()


class TrainSSLCB(sh.callbacks.Callback):
    def __init__(self, ssl_dl, logger=None): 
        sh.utils.store_attr(self, locals())
        #self.scaler = torch.cuda.amp.GradScaler() 
        self.epoch_start_ssl = 40
        self.ssl_criterion = torch.nn.BCEWithLogitsLoss()
        self.ssl_loss_scale = 2

    def before_fit(self): 
        assert self.kwargs['cfg'].TRAIN.AMP
        #import torch_ema
        #self.ema = torch_ema.ExponentialMovingAverage(self.model.parameters(), decay=0.995)


    @sh.utils.on_train
    def before_epoch(self):
        if self.kwargs['cfg'].PARALLEL.DDP: self.dl.sampler.set_epoch(self.n_epoch)
        self.ssl_it = iter(self.ssl_dl)

        for i in range(len(self.opt.param_groups)):
            self.learner.opt.param_groups[i]['lr'] = self.lr  

    def train_step(self):
        xb, yb = self.batch

        with torch.cuda.amp.autocast(enabled=False):
            self.learner.preds = self.model(xb)
            loss1 = self.loss_func(self.learner.preds, yb)

            ssl_loss = None
            if self.n_epoch >= self.epoch_start_ssl:
                try:
                    xbs, ybs = next(self.ssl_it)
                    xbs = xbs.cuda()
                    with torch.no_grad():
                        #self.ema.update(self.model.parameters())
                        ybs = self.model(ybs.cuda()).sigmoid()
                        mask = (ybs < .1 ) +  (ybs > .8) # TODO blur? ema?
                        ybs_m = ybs[mask]

                    preds = self.model(xbs)
                    preds_m = preds[mask]
                    if self.np_batch > .9 and np.random.random() > .5: 
                        #print(xbs.shape, ybs.shape, preds.shape)
                        self.learner.batch = (xbs,ybs)
                        self.learner.preds = preds

                    ssl_loss = self.ssl_criterion(preds_m, ybs_m) * self.ssl_loss_scale

                except StopIteration:
                    print('maxit:', self.np_batch)
                    pass

            if ssl_loss is not None:
                print(loss1.item(), ssl_loss.item())
                loss = (loss1 + ssl_loss) / 2
            else:
                loss = loss1

            self.learner.loss = loss
            
        loss.backward()
        self.learner.opt.step()
        #self.scaler.scale(loss).backward()
        #self.scaler.step(self.learner.opt)
        #self.scaler.update()

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
        self.reduction = 'none' if self.kwargs['cfg'].TRAIN.SELECTIVE_BP <1 - 1e-6 else 'mean'

    def run_extra_valid(self):
        self.learner.extra_accs, self.learner.extra_samples_count = [], []
        for batch in self.extra_valid_dl:
            xb, yb = get_xb_yb(batch)
            tag = get_tag(batch)
            batch_size = xb.shape[0]

            with torch.no_grad():
                preds = self.model(xb)

            #print(preds.shape, yb.shape, xb.shape)
            p = torch.sigmoid(preds.cpu().float())
            p = (p>.4).float()
            dice = loss.dice_loss(p, yb.cpu().float())
            #dice = loss.dice_loss(torch.sigmoid(preds.cpu().float()), yb.cpu().float())
            #print(n, dice, dice*n)
            self.learner.extra_accs.append(dice * batch_size)
            self.learner.extra_samples_count.append(batch_size)

    def val_step(self):
        xb, yb = self.batch
        with torch.no_grad():
            #with torch.cuda.amp.autocast():
            self.learner.preds = self.model(xb)
            self.learner.loss = self.loss_func(self.preds, yb, reduction=self.reduction)
                

