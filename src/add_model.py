from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel


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
