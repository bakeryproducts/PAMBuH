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

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.in_channels == 3: nn.init.normal_(m.weight, 0, 0.21)
            if isinstance(m, nn.Conv2d) : nn.init.kaiming_uniform_(m.weight, a=.1, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.ConvTranspose2d): nn.init.kaiming_uniform_(m.weight, a=.1, mode='fan_in', nonlinearity='leaky_relu')#nn.init.xavier_uniform_(m.weight, 0.1)
            if m.bias is not None: m.bias.data.zero_()
    if hasattr(model, '_layer_init'):
        model._layer_init()

def model_select():
    #model = smp.manet.MAnet
    #model = smp.Unet

    model = partial(smp.manet.MAnet, encoder_name='timm-efficientnet-b4')
    #model = smp.DeepLabV3Plus
    #model = partial(smp.manet.MAnet, encoder_name='resnet50')
    #model = partial(smp.UnetPlusPlus, encoder_name='se_resnet50')#, encoder_depth=4, decoder_channels=[256,128,32,16])
    return model

def build_model(cfg):
    base_lr = 1e-4# should be overriden in LR scheduler anyway
    lr = base_lr if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    
    model = model_select()()
    if cfg.TRAIN.INIT_MODEL: 
        logger.log('DEBUG', f'Init model: {cfg.TRAIN.INIT_MODEL}') 
        model = _load_model_state(model, cfg.TRAIN.INIT_MODEL)
    else:
        pass
        #init_model(model)
        #nn.init.constant_(model.segmentation_head[0].bias, -4.59) # manet

    model = model.cuda()
    
    opt = optim.AdamW
    opt_kwargs = {}
    optimizer = opt(tencent_trick(model), lr=lr, **opt_kwargs)
    
    if cfg.PARALLEL.DDP: model = DistributedDataParallel(model, device_ids=[cfg.PARALLEL.LOCAL_RANK], find_unused_parameters=True)
    model.train()
    return model, optimizer

def load_model(cfg, model_folder_path, eval_mode=True):
    print(model_folder_path)
    # model_select syncing build and load, probably should be in cfg, by key as in datasets
    model = model_select()()
    model = _load_model_state(model, model_folder_path)
    if eval_mode: model.eval()
    return model

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

def parse_model_path(p):
    name = str(p.name)
    epoch = name.split('_')[0]
    return int(epoch[1:])

def get_last_model_name(src):
    # assumes that model name is of type e500_blabla.pth, sorted by epoch #500
    model_names = list(Path(src).glob('*.pth'))
    assert model_names != [], 'No valid models at init path'

    res = []
    for i, m in enumerate(model_names):
        epoch = parse_model_path(m)
        res.append([i,epoch])
    idx = sorted(res, key=lambda x: -x[1])[0][0]
    return model_names[idx]
