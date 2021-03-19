from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel



class FoldModel(nn.Module):
    def __init__(self, models):
        super(FoldModel, self).__init__()
        self.ms = models
        
    def forward(self, x):
        res = torch.stack([m(x) for m in self.ms])
        return res.mean(0)

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

class conv_once(nn.Module):
    '''conv BN lrelu'''
    def __init__(self,fi,fo, name='convbnr', slope=.1):
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
    def __init__(self, fi, fo, name='convbnr2', slope=.1):
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
    def __init__(self, fi, fo, name='up', slope=.1):
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
    def __init__(self,fi, fo, name='cndown', slope=.1):
        super(downsample_block, self).__init__()
        name += '_'
        self.block = nn.Sequential(
                OrderedDict([
                        (name+'conv', nn.Conv2d(fi, fo, kernel_size=3,stride=2,padding=1,bias=False)),
                        (name+'norm', nn.BatchNorm2d(fo)),
                        (name+'relu', nn.LeakyReLU(slope, inplace=True)),
                ]))
    def forward(self, x): return self.block(x)

class __UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block(in_channels, nb_filter[0])
        self.conv1_0 = conv_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = conv_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = conv_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = conv_block(nb_filter[3], nb_filter[4])

        self.conv3_1 = conv_block(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = conv_block(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = conv_block(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = conv_block(nb_filter[0]+nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)


    def _layer_init(self): nn.init.constant_(self.final.bias, -4.59)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class seg_attention_block(nn.Module):
    """https://arxiv.org/pdf/1804.03999.pdf"""
    def __init__(self, fg, fl, fint, name='att'):
        super(seg_attention_block, self).__init__()
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


class base_enc_unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(base_enc_unet, self).__init__()
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
        
    def cat(self, a,b):
        # cat with fixed shape
        min_x = min(a.shape[-2], b.shape[-2])
        min_y = min(a.shape[-1], b.shape[-1])
        return torch.cat((a[...,:min_x, :min_y], b[...,:min_x, :min_y]), dim=1)
        
    def forward_encoding(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.downsample1(e1))
        e3 = self.e3(self.downsample2(e2))
        e4 = self.e4(self.downsample3(e3))
        bottleneck = self.bottleneck(self.downsample4(e4))
        return e1,e2,e3,e4,bottleneck

    
class base_unet(base_enc_unet):
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

    def forward(self, x):
        e1,e2,e3,e4,bottleneck = self.forward_encoding(x)
        
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


class Unet(base_unet):
    def __init__(self, *args, **kwargs):
        super(Unet, self).__init__( *args, **kwargs)
        self.conv_1x1 = nn.Conv2d(self.init_features, self.out_channels, kernel_size=1,stride=1,padding=0)
    
    def _layer_init(self): nn.init.constant_(self.conv_1x1.bias, -4.59)
    
    def forward(self, x):
        x = super(Unet, self).forward(x)
        return self.conv_1x1(x)


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
