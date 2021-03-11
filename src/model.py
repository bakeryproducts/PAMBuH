from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel

from add_model import *


class NestedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, deep_supervision=True, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.downsample1 = conv_block(nb_filter[0], nb_filter[1])
        self.downsample2 = conv_block(nb_filter[1], nb_filter[2])
        self.downsample3 = conv_block(nb_filter[2], nb_filter[3])
        self.downsample4 = conv_block(nb_filter[3], nb_filter[4])

        self.up = nn.Upsample(scale_factor=2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #self.upsample1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        #self.upsample2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        #self.upsample3 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)
        #self.upsample4 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2)

        self.conv0_0 = conv_block(in_channels,  nb_filter[0]) # TODO stem

        self.conv0_1 = conv_block(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = conv_block(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = conv_block(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = conv_block(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = conv_block(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = conv_block(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = conv_block(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = conv_block(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = conv_block(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = conv_block(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def _layer_init(self): 
        if not self.deep_supervision:
            nn.init.constant_(self.final.bias, -4.59)
        else:
            nn.init.constant_(self.final1.bias, -4.59)
            nn.init.constant_(self.final2.bias, -4.59)
            nn.init.constant_(self.final3.bias, -4.59)
            nn.init.constant_(self.final4.bias, -4.59)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.downsample1(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.downsample2(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        
        x3_0 = self.downsample3(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.downsample4(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            out = torch.mean(torch.cat([output1, output2, output3, output4], 1), 1, keepdim=True)
            #print(out.shape)
            return out
        else:
            output = self.final(x0_4)
            return output


class att_unet(base_unet):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(att_unet, self).__init__(in_channels, out_channels, init_features)
        f = self.init_features
        self.ag4 = seg_attention_block(f*8, f*8, f*4, 'att4')
        self.ag3 = seg_attention_block(f*4, f*4, f*2, 'att3')
        self.ag2 = seg_attention_block(f*2, f*2, f, 'att2')
        self.ag1 = seg_attention_block(f, f, f, 'att1')
        self.conv_1x1 = nn.Conv2d(f, self.out_channels, kernel_size=1,stride=1,padding=0)
    
    def _layer_init(self): nn.init.constant_(self.conv_1x1.bias, -4.59)
    
    def forward(self, x):
        e1,e2,e3,e4,bottleneck = self.forward_encoding(x)
         
        d4 = self.upsample4(bottleneck) 
        ea4 = self.ag4(d4, e4)
        d4 = self.d4(self.cat(d4,ea4))
        
        d3 = self.upsample3(d4)
        ea3 = self.ag3(d3, e3)
        d3 = self.d3(self.cat(d3,ea3))
        
        d2 = self.upsample2(d3) 
        ea2 = self.ag2(d2, e2)
        d2 = self.d2(self.cat(d2,ea2))
        
        d1 = self.upsample1(d2)
        ea1 = self.ag1(d1, e1)
        d1 = self.d1(self.cat(d1,ea1))
        
        return self.conv_1x1(d1)


class Unet(base_unet):
    def __init__(self, *args, **kwargs):
        super(Unet, self).__init__( *args, **kwargs)
        self.conv_1x1 = nn.Conv2d(self.init_features, self.out_channels, kernel_size=1,stride=1,padding=0)
    
    def _layer_init(self): nn.init.constant_(self.conv_1x1.bias, -4.59)
    
    def forward(self, x):
        x = super(Unet, self).forward(x)
        return self.conv_1x1(x)

class Unet_x05(base_unet):
    def __init__(self, *args, **kwargs):
        super(Unet_x05, self).__init__( *args, **kwargs)
        f = self.init_features
        self.downsample21 = downsample_block(f,f,'d21')
        self.e21 = conv_block(f, f, 'e21')
        self.conv_1x1 = nn.Conv2d(f, self.out_channels, kernel_size=1,stride=1,padding=0)
    
    def _layer_init(self): nn.init.constant_(self.conv_1x1.bias, -4.59)
    
    def forward(self, x):
        x = super(Unet_x05, self).forward(x)
        e21 = self.e21(self.downsample21(x))
        return self.conv_1x1(e21)
    
class Unet_x025(base_unet):
    def __init__(self, *args, **kwargs):
        super(Unet_x025, self).__init__( *args, **kwargs)
        f = self.init_features
        self.downsample21 = downsample_block(f,f,'d21')
        self.e21 = conv_block(f, 2*f, 'e21')
        self.downsample22 = downsample_block(2*f,2*f,'d22')
        self.e22 = conv_block(2*f, 2*f, 'e22')
        self.conv_1x1 = nn.Conv2d(f, self.out_channels, kernel_size=1,stride=1,padding=0)
    
    def _layer_init(self): nn.init.constant_(self.conv_1x1.bias, -4.59)
    
    def forward(self, x):
        x = super(Unet_x025, self).forward(x)
        e21 = self.e21(self.downsample21(x))
        e22 = self.e22(self.downsample22(e21))
        return self.conv_1x1(e22)


class domain_head(nn.Module):
    def __init__(self, in_channels=16*32):
        super(base_enc_unet, self).__init__()
        f = in_channels 
        self.fc1 = torch.nn.Linear(f, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (2*x).detach() - x
        x = self.fc1(self.relu(x))
        x = self.fc2(self.relu(x))
        return x

class domain_unet(base_enc_unet):
    def __init__(self, *args, **kwargs):
        super(base_unet, self).__init__( *args, **kwargs)
        f = self.init_features
        
        self.upsample4 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.d4 =  conv_block(16*f, 8*f, 'd4')
        self.upsample3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.d3 =  conv_block(8*f, 4*f, 'd3')
        self.upsample2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.d2 =  conv_block(4*f, 2*f, 'd2')
        self.upsample1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.d1 =  conv_block(2*f, f, 'd1')

        self.dhead = self.domain_head(f*32) # 32 for encoder deepest level
        self.conv_1x1 = nn.Conv2d(self.init_features, self.out_channels, kernel_size=1,stride=1,padding=0)

    def _layer_init(self): nn.init.constant_(self.conv_1x1.bias, -4.59)

    def forward(self, x):
        e1,e2,e3,e4,bottleneck = self.forward_encoding(x)
        domain = self.dhead(bottleneck)
        
        # decoding + concat path
        d4 = self.upsample4(bottleneck) 
        d4 = self.d4(self.cat(d4,e4))
        
        d3 = self.upsample3(d4)
        d3 = self.d3(self.cat(d3,e3))
        
        d2 = self.upsample2(d3) 
        d2 = self.d2(self.cat(d2,e2))
        
        d1 = self.upsample1(d2)
        d1 = self.d1(self.cat(d1,e1))

        return self.conv_1x1(d1),  domain



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
            if m.in_channels == 3: nn.init.normal_(m.weight, 0, 0.21)
            if isinstance(m, nn.Conv2d) : nn.init.kaiming_uniform_(m.weight, a=.1, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.ConvTranspose2d): nn.init.kaiming_uniform_(m.weight, a=.1, mode='fan_in', nonlinearity='leaky_relu')#nn.init.xavier_uniform_(m.weight, 0.1)
            if m.bias is not None: m.bias.data.zero_()
    model._layer_init()


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]


def model_select():
    return Unet
    #return NestedUNet
    #return att_unet

def build_model(cfg):
    base_lr = 1e-4# should be overriden in LR scheduler anyway
    lr = base_lr if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    
    model = model_select()(in_channels=3, out_channels=1)
    init_model(model)
    model = model.cuda()
    
    opt = optim.AdamW
    #opt = optim.SGD
    #opt_kwargs = {'momentum':0.9, 'weight_decay':1e-2}
    opt_kwargs = {}
    optimizer = opt(tencent_trick(model), lr=lr, **opt_kwargs)
    
    if cfg.PARALLEL.DDP: model = DistributedDataParallel(model, device_ids=[cfg.PARALLEL.LOCAL_RANK])
    model.train()
    return model, optimizer

def load_model(cfg, path):
    # model_select syncing build and load, probably should be in cfg, by key as in datasets
    model = model_select()(in_channels=3, out_channels=1)
    
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

class FoldModel(nn.Module):
    def __init__(self, models):
        super(FoldModel, self).__init__()
        self.ms = models
        
    def forward(self, x):
        res = torch.stack([m(x) for m in self.ms])
        return res.mean(0)
