import os
import gc
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
from fastprogress.fastprogress import master_bar, progress_bar

import shallow as sh
import data
import loss
import utils
from logger import logger
from config import cfg, cfg_init
from model import build_model
from callbacks import *


def clo(logits, predicts):
    w1 = .2
    w2 = 1 - w1 
    l1 = loss.lovasz_hinge(logits, predicts) 
    l2 = torch.nn.functional.binary_cross_entropy_with_logits(logits, predicts)
    return l1*w1 + l2*w2

def start(cfg, output_folder):
    datasets = data.build_datasets(cfg, dataset_types=['TRAIN', 'VALID'])
    n = cfg.TRAIN.NUM_FOLDS
    if n <= 1: start_fold(cfg, output_folder, datasets)
    else: 
        datasets_folds = data.make_datasets_folds(cfg, datasets, n, shuffle=True)
        for i, i_datasets in enumerate(datasets_folds):
            if cfg.PARALLEL.IS_MASTER: print(f'\n\nFOLD # {i}\n\n')
            fold_output = None
            if output_folder:
                fold_output = output_folder / f'fold_{i}'
                fold_output.mkdir()
            start_fold(cfg, fold_output, i_datasets)
            if cfg.PARALLEL.IS_MASTER: print(f'\n\n END OF FOLD # {i}\n\n')


def start_fold(cfg, output_folder, datasets):
    n_epochs = cfg.TRAIN.EPOCH_COUNT
    dls = data.build_dataloaders(cfg, datasets, pin=True, drop_last=True)
    model, opt = build_model(cfg)

    criterion = clo
    train_cb = TrainCB(logger=logger)
    val_cb = ValCB(logger=logger)
    
    if cfg.PARALLEL.IS_MASTER:
        utils.dump_params(cfg, output_folder)
        models_dir = output_folder / 'models'
        tb_dir = output_folder / 'tb'
        models_dir.mkdir()
        tb_dir.mkdir()

        step = cfg.TRAIN.TB_STEP
        writer = SummaryWriter(log_dir=tb_dir, comment='Demo')
        
        tb_metric_cb = partial(TBMetricCB, 
                               writer=writer,
                               train_metrics={'losses':['train_loss'], 'general':['lr', 'train_dice']}, 
                               validation_metrics={'losses':['val_loss'], 'general':['valid_dice', 'valid_dice2']})

        tb_predict_cb = partial(TBPredictionsCB, writer=writer, logger=logger, step=step)

        tb_cbs = [tb_metric_cb(), tb_predict_cb()]
        checkpoint_cb = sh.callbacks.CheckpointCB(models_dir, save_step=cfg.TRAIN.SAVE_STEP)
        train_timer_cb = sh.callbacks.TimerCB(mode_train=True, logger=logger)
        master_cbs = [train_timer_cb, *tb_cbs, checkpoint_cb]
    
    l0,l1,l2 = 5e-5, 2e-4,5e-5
    scale = 1 #1/ cfg.PARALLEL.WORLD_SIZE

    l0,l1,l2 = l0/scale, l1/scale, l2/scale
    lr_cos_sched = sh.schedulers.combine_scheds([
        [.075, sh.schedulers.sched_cos(l0,l1)],
        [.925, sh.schedulers.sched_cos(l1,l2)]])
    lrcb = sh.callbacks.ParamSchedulerCB('before_epoch', 'lr', lr_cos_sched)
    cbs = [CudaCB(), train_cb, val_cb, lrcb]
        
    if cfg.PARALLEL.IS_MASTER:
        cbs.extend(master_cbs)
        # TODO nfold in epochbar
        epoch_bar = master_bar(range(n_epochs))
        batch_bar = partial(progress_bar, parent=epoch_bar)
    else: epoch_bar, batch_bar = range(n_epochs), lambda x:x
    logger.log("DEBUG", 'LEARNER') 

    val_interval = cfg.VALID.STEP

    learner = sh.learner.Learner(model, opt, sh.utils.AttrDict(dls), criterion, 0, cbs, batch_bar, epoch_bar, val_interval, cfg=cfg)
    learner.fit(n_epochs)

    gc.collect()
    torch.cuda.empty_cache()
    del dls 
    del learner
    del cbs
    gc.collect()
    torch.cuda.empty_cache()
