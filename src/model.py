from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init



class conv_once(nn.Module):
    '''conv BN lrelu'''
    def __init__(self,fi,fo, name, slope=.1):
        super(conv_once, self).__init__()
        name += '_'
        self.block = nn.Sequential(
                OrderedDict([
                        (name+'conv', nn.Conv2d(fi, fo, kernel_size=3,stride=1,padding=1,bias=False)),
                        (name+'norm', nn.BatchNorm2d(fo)),
                        (name+'relu', nn.LeakyReLU(slope, inplace=True)),
                ]))
    def forward(self, x): return self.block(x)

            
class conv_block(nn.Module):
    '''conv BN lrelu x2'''
    def __init__(self, fi, fo, name, slope=.1):
        super(conv_block, self).__init__()
        name += '_'
        self.block = nn.Sequential(
                OrderedDict([
                        (name+'conv_block_1', conv_once(fi,fo,name+'1', slope)),
                        (name+'conv_block_2', conv_once(fo,fo,name+'2', slope)),
                ]))
            
    def forward(self, x): return self.block(x)

class upsample_block(nn.Module):
    ''' spatial 2x '''
    def __init__(self, fi, fo, name, slope=.1):
        super(upsample_block, self).__init__()
        name += '_'
        self.block = nn.Sequential(
                OrderedDict([
                        (name+'upsample', nn.Upsample(scale_factor=2)),
                        (name+'conv_block_1', conv_once(fi,fo,name+'1', slope))
                ]))
            
    def forward(self,x): return self.block(x)
            
class downsample_block(nn.Module):
    '''conv BN lrelu'''
    def __init__(self,fi, fo, name, slope=.1):
        super(downsample_block, self).__init__()
        name += '_'
        self.block = nn.Sequential(
                OrderedDict([
                        (name+'conv', nn.Conv2d(fi, fo, kernel_size=3,stride=2,padding=1,bias=False)),
                        (name+'norm', nn.BatchNorm2d(fo)),
                        (name+'relu', nn.LeakyReLU(slope, inplace=True)),
                ]))
    def forward(self, x): return self.block(x)

class seg_attention_block(nn.Module):
    """https://arxiv.org/pdf/1804.03999.pdf"""
    def __init__(self, fg, fl, fint):
        super(Attention_block, self).__init__()
        name1 = name +'_gate_'
        self.wg = nn.Sequential(OrderedDict([
                                (name1+'conv',nn.Conv2d(fg, fint, kernel_size=1,stride=1,padding=0,bias=False)),
                                (name1+'norm',nn.BatchNorm2d(fint))
        ]))
                                
        name2 = name +'_signal_'             
        self.wx = nn.Sequential(OrderedDict([
                                (name2+'conv',nn.Conv2d(fl, fint, kernel_size=1,stride=1,padding=0,bias=False)),
                                (name2+'norm',nn.BatchNorm2d(fint))
        ]))
            
        name3 = name +'_merge_'
        self.psi = nn.Sequential(OrderedDict([
                                (name3+'conv',nn.Conv2d(fint, 1, kernel_size=1,stride=1,padding=0,bias=False)),
                                (name3+'norm',nn.BatchNorm2d(1)),
                                (name3+'sigmoid', nn.Sigmoid())
        ]))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.wg(g)
        x1 = self.wx(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


class _base_unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(_base_unet, self).__init__()
        self.init_features = init_features
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        f = init_features
        self.e1 = conv_block(in_channels, f, 'e1')
        self.downsample1 = downsample_block(f,f,'d1')
        self.e2 = conv_block(f, 2*f, 'e2')
        self.downsample2 = downsample_block(2*f,2*f,'d2')
        self.e3 = conv_block(2*f, 4*f, 'e3')
        self.downsample3 = downsample_block(4*f,4*f,'d3')
        self.e4 = conv_block(4*f, 8*f, 'e4')
        self.downsample4 = downsample_block(8*f,8*f,'d4')
        self.bottleneck = conv_block(8*f, 16*f, 'bottleneck')
        
        self.upsample4 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.d4 =  conv_block(16*f, 8*f, 'd4')
        self.upsample3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.d3 =  conv_block(8*f, 4*f, 'd3')
        self.upsample2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.d2 =  conv_block(4*f, 2*f, 'd2')
        self.upsample1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.d1 =  conv_block(2*f, f, 'd1')

    def _layer_init(self): nn.init.constant_(self.conv_1x1.bias, -4.59)
 
    def cat(self, a,b):
        # cat with fixes shape
        min_x = min(a.shape[-2], b.shape[-2])
        min_y = min(a.shape[-1], b.shape[-1])
        return torch.cat((a[...,:min_x, :min_y], b[...,:min_x, :min_y]), dim=1)

    def forward(self,x):
        # encoding path
        e1 = self.e1(x)
        e2 = self.e2(self.downsample1(e1))
        e3 = self.e3(self.downsample2(e2))
        e4 = self.e4(self.downsample3(e3))
        bottleneck = self.bottleneck(self.downsample4(e4))
        
        # decoding + concat path
        d4 = self.upsample4(bottleneck) 
        d4 = self.d4(self.cat(d4,e4))
        
        d3 = self.upsample3(d4) 
        d3 = self.d3(self.cat(d3,e3))
        
        d2 = self.upsample2(d3) 
        d2 = self.d2(self.cat(d2,e2))
        
        d1 = self.upsample1(d2) 
        d1 = self.d1(self.cat(d1,e1))
        
        return d1


class Unet(_base_unet):
    def __init__(self, *args, **kwargs):
        super(Unet, self).__init__( *args, **kwargs)
        self.conv_1x1 = nn.Conv2d(self.init_features, self.out_channels, kernel_size=1,stride=1,padding=0)
    
    def forward(self, x):
        x = super(Unet, self).forward(x)
        return self.conv_1x1(x)

class Unet_x05(_base_unet):
    def __init__(self, *args, **kwargs):
        super(Unet_x05, self).__init__( *args, **kwargs)
        f = self.init_features
        self.downsample21 = downsample_block(f,f,'d21')
        self.e21 = conv_block(f, f, 'e21')
        self.conv_1x1 = nn.Conv2d(f, self.out_channels, kernel_size=1,stride=1,padding=0)
    
    def forward(self, x):
        x = super(Unet_x05, self).forward(x)
        e21 = self.e21(self.downsample21(x))
        return self.conv_1x1(e21)
    
class Unet_x025(_base_unet):
    def __init__(self, *args, **kwargs):
        super(Unet_x025, self).__init__( *args, **kwargs)
        f = self.init_features
        self.downsample21 = downsample_block(f,f,'d21')
        self.e21 = conv_block(f, 2*f, 'e21')
        self.downsample22 = downsample_block(2*f,2*f,'d22')
        self.e22 = conv_block(2*f, 2*f, 'e22')
        self.conv_1x1 = nn.Conv2d(f, self.out_channels, kernel_size=1,stride=1,padding=0)
    
    def forward(self, x):
        x = super(Unet_x025, self).forward(x)
        e21 = self.e21(self.downsample21(x))
        e22 = self.e22(self.downsample22(e21))
        return self.conv_1x1(e22)

def to_cuda(models): return [model.cuda() for model in models]
def scale_lr(lr, cfg): return lr * float(cfg.TRAIN.BATCH_SIZE * cfg.PARALLEL.WORLD_SIZE)/256.
def sync_bn(models): return [apex.parallel.convert_syncbn_model(m) for m in models]
def get_trainable_parameters(model): return model.parameters()
def get_lr(**kwargs): return kwargs.get('lr', 1e-4)
def dd_parallel(models): return [DistributedDataParallel(m) for m in models]

def autocaster(foo, *args, **kwargs):
    with torch.cuda.amp.autocast():
        return foo(*args, **kwargs)

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.in_channels == 3: nn.init.normal_(m.weight, 0,.21)
            elif isinstance(m, nn.Conv2d) : nn.init.kaiming_uniform_(m.weight, a=.1, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.ConvTranspose2d): nn.init.kaiming_uniform_(m.weight, a=.1, mode='fan_in', nonlinearity='leaky_relu')#nn.init.xavier_uniform_(m.weight, 0.1)
            if m.bias is not None: m.bias.data.zero_()
    model._layer_init()


def build_model(cfg):
    base_lr = 1e-4# should be overriden in LR scheduler anyway
    lr = base_lr if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    
    model = Unet(in_channels=3, out_channels=1)
    init_model(model)
    
    opt = optim.AdamW
    opt_kwargs = {}
    optimizer = opt(get_trainable_parameters(model), lr=lr, **opt_kwargs)
    
    if cfg.PARALLEL.DDP: model = DistributedDataParallel(model, device_ids=[cfg.PARALLEL.LOCAL_RANK])
    model.train()
    
    return model, optimizer

def load_model(path):
    raise NotImplementedError
