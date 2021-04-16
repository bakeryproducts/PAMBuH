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
    #model = partial(smp.Unet, encoder_name='se_resnet50')
    #model = partial(smp.UnetPlusPlus, encoder_name='se_resnet50')
    #model = smp.PAN
#['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
# 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d',
# 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d',
# 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
# 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
# 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8',
# 'timm-efficientnet-l2',
# 'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d',
# 'timm-resnest50d_1s4x24d',
# 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s', 'timm-res2next50',
# 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032',
# 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320',
# 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032',
# 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 'timm-regnety_320',
# 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d']

    #model = smp.MAnet
    #model = partial(smp.manet.MAnet, encoder_name='timm-efficientnet-b4')
    #model = partial(smp.manet.MAnet, encoder_name='resnext50_32x4d')
    #model = partial(smp.manet.MAnet, encoder_name='timm-regnetx_032')
    #model = partial(smp.manet.MAnet, encoder_name='resnet50')
    #model = partial(smp.manet.MAnet, encoder_name='se_resnet50')

    #model = partial(smp.UnetPlusPlus, encoder_name='se_resnet50',aux_params={'classes':1, 'activation':None, 'dropout':0}) 
    #model = partial(smp.UnetPlusPlus, encoder_name='timm-regnetx_032')

    #model = partial(smp.Unet, encoder_name='timm-regnetx_032', aux_params={'classes':1, 'activation':None, 'dropout':0})
    #model = partial(smp.Unet, encoder_name='timm-regnety_032', aux_params={'classes':1, 'activation':None, 'dropout':0})

    model = partial(smp.Unet, encoder_name='timm-regnetx_032')
    #model = partial(smp.Unet, encoder_name='timm-efficientnet-b2')
    #model = partial(smp.Unet, encoder_name='resnet18')
    #model = partial(smp.Unet)

    #model = partial(smp.Unet, encoder_name='timm-regnetx_064')
    #model = partial(smp.Unet, encoder_name='resnet34', decoder_use_batchnorm=True)

    #model = partial(smp.manet.MAnet, encoder_weights='ssl', encoder_name='resnext101_32x4d')
    #model = partial(smp.manet.MAnet,  encoder_weights=None)

    #model = smp.DeepLabV3Plus
    #model = partial(smp.DeepLabV3Plus, encoder_name='timm-regnetx_032')
    #model = partial(smp.UnetPlusPlus, encoder_name='timm-efficientnet-b4')#, encoder_depth=4, decoder_channels=[256,128,32,16])
    return model

def build_model(cfg):
    model = model_select()()
    if cfg.TRAIN.INIT_MODEL: 
        logger.log('DEBUG', f'Init model: {cfg.TRAIN.INIT_MODEL}') 
        model = _load_model_state(model, cfg.TRAIN.INIT_MODEL)
    else: pass
    init_model(model)
    model = model.cuda()
    model.train()
    return model 

def get_optim(cfg, model):
    base_lr = 1e-4# should be overriden in LR scheduler anyway
    lr = base_lr if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    
    #if cfg.PARALLEL.DDP:   
    #    optimizer = ZeroRedundancyOptimizer(tencent_trick(model), optim=optim.AdamW, lr=lr)
    #else:
    opt = optim.AdamW
    opt_kwargs = {}
    optimizer = opt(tencent_trick(model), lr=lr, **opt_kwargs)

    return optimizer

def wrap_ddp(cfg, model):
    if cfg.PARALLEL.DDP: 
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, 
                                    device_ids=[cfg.PARALLEL.LOCAL_RANK],
                                    find_unused_parameters=False,
                                    broadcast_buffers=True)
    return model


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
