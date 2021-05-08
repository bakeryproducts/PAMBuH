from pathlib import Path
from functools import partial
from collections import OrderedDict
from logger import logger

import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import segmentation_models_pytorch as smp

from add_model import *

def to_cuda(models): return [model.cuda() for model in models]
def scale_lr(lr, cfg): return lr * float(cfg.TRAIN.BATCH_SIZE * cfg.PARALLEL.WORLD_SIZE)/256.
def sync_bn(models): return [apex.parallel.convert_syncbn_model(m) for m in models]
def get_trainable_parameters(model): return model.parameters()
def get_lr(**kwargs): return kwargs.get('lr', 1e-4)
def dd_parallel(models): return [DistributedDataParallel(m) for m in models]


def model_select():
    #model = partial(smp.Unet, encoder_name='timm-regnetx_032')
    #model = partial(smp.Unet, encoder_name='timm-regnetx_032', encoder_weights=None)
    #model = partial(smp.Unet, encoder_name='timm-efficientnet-b2')
    #model = partial(smp.Unet, encoder_name='timm-regnetx_032', aux_params={'classes':1, 'activation':None, 'dropout':0}) 
    #model = partial(smp.Unet, encoder_name='timm-regnetx_032') 
    #model = partial(smp.Unet, encoder_name='timm-regnetx_032')

    #model = partial(smp.Unet, encoder_name='timm-regnetx_032')
    model = partial(smp.Unet, encoder_name='timm-regnety_016')
    #model = partial(smp.Linknet, encoder_name='timm-regnety_016')
    #model = partial(smp.FPN, encoder_name='timm-regnety_016')
    #model = partial(smp.PAN, encoder_name='timm-regnety_016')
    #model = partial(smp.UnetPlusPlus, encoder_name='timm-regnety_016')
    #model = partial(smp.Unet, encoder_name='se_resnet50')
    

    #model = partial(smp.UnetPlusPlus, encoder_name='timm-regnety_008')
    #model = partial(smp.Linknet, encoder_name='timm-regnety_016')
    #model = partial(smp.Unet, encoder_name='se_resnet50')
    #model = partial(smp.Unet)
    #model = partial(smp.Unet, encoder_name='resnet34', decoder_use_batchnorm=True)
    return model

def build_model(cfg):
    model = model_select()()
    #replace_relu_to_silu(model)
    if cfg.TRAIN.INIT_MODEL: 
        logger.log('DEBUG', f'Init model: {cfg.TRAIN.INIT_MODEL}') 
        model = _load_model_state(model, cfg.TRAIN.INIT_MODEL)
    else: pass

    model = model.cuda()
    model.train()
    return model 

def get_optim(cfg, model):
    base_lr = 1e-4# should be overriden in LR scheduler anyway
    lr = base_lr if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    
    #opt = optim.AdamW
    opt = optim.Adam
    #opt_kwargs = {'amsgrad':True, 'weight_decay':1e-3}
    opt_kwargs = {}
    optimizer = opt(tencent_trick(model), lr=lr, **opt_kwargs)
    if cfg.TRAIN.INIT_MODEL: 
        st =  _load_opt_state(model, cfg.TRAIN.INIT_MODEL)
        optimizer.load_state_dict(st)
    return optimizer

def wrap_ddp(cfg, model):
    if cfg.PARALLEL.DDP: 
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, 
                                    device_ids=[cfg.PARALLEL.LOCAL_RANK],
                                    find_unused_parameters=True,
                                    broadcast_buffers=True)
    return model


def load_model(cfg, model_folder_path, eval_mode=True):
    print(model_folder_path)
    # model_select syncing build and load, probably should be in cfg, by key as in datasets
    model = model_select()()
    model = _load_model_state(model, model_folder_path)
    if eval_mode: model.eval()
    return model

def _load_opt_state(model, path):
    path = Path(path)
    if path.suffix != '.pth': path = get_last_model_name(path)
    opt_state = torch.load(path)['opt_state']
    return opt_state

def _load_model_state(model, path):
    path = Path(path)
    if path.suffix != '.pth': path = get_last_model_name(path)
    state_dict = torch.load(path)['model_state']

    # Strip ddp model TODO dont save it like that
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            k = k.lstrip('module')[1:]
        new_state_dict[k] = v
    del state_dict
    
    model.load_state_dict(new_state_dict)
    del new_state_dict
    return model

