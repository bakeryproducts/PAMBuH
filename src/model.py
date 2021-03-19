from collections import OrderedDict
from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel

from add_model import *



class Unet(base_unet):
    def __init__(self, *args, **kwargs):
        super(Unet, self).__init__( *args, **kwargs)
        self.conv_1x1 = nn.Conv2d(self.init_features, self.out_channels, kernel_size=1,stride=1,padding=0)
    
    def _layer_init(self): nn.init.constant_(self.conv_1x1.bias, -4.59)
    
    def forward(self, x):
        x = super(Unet, self).forward(x)
        return self.conv_1x1(x)


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
    #model = Unet
    #model = NestedUNet
    import segmentation_models_pytorch as smp
    model = smp.manet.MAnet
    #model = partial(smp.manet.MAnet, encoder_name='timm-efficientnet-b3')
    #model = partial(smp.manet.MAnet, encoder_name='resnet50')
    #model = partial(smp.UnetPlusPlus, encoder_name='se_resnet50')#, encoder_depth=4, decoder_channels=[256,128,32,16])
    return model

def build_model(cfg):
    base_lr = 1e-4# should be overriden in LR scheduler anyway
    lr = base_lr if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    
    model = model_select()()
    #init_model(model)
    #nn.init.constant_(model.segmentation_head[0].bias, -4.59) # manet
    model = model.cuda()
    
    opt = optim.AdamW
    opt_kwargs = {}
    optimizer = opt(tencent_trick(model), lr=lr, **opt_kwargs)
    
    if cfg.PARALLEL.DDP: model = DistributedDataParallel(model, device_ids=[cfg.PARALLEL.LOCAL_RANK], find_unused_parameters=True)
    model.train()
    return model, optimizer

def load_model(cfg, path):
    # model_select syncing build and load, probably should be in cfg, by key as in datasets
    model = model_select()()
    
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
    
    model.eval()
    return model

