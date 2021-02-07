from functools import partial

import albumentations as albu
import albumentations.pytorch as albu_pt
import torch  # to tensor transform
import numpy as np


class ToTensor(albu_pt.ToTensorV2):
    def apply_to_mask(self, mask, **params): return torch.from_numpy(mask).permute((2,0,1))
    def apply(self, image, **params): return torch.from_numpy(image).permute((2,0,1))

class Augmentator:
    def __init__(self, cfg):
        self.cfg = cfg 
        self.resize_w, self.resize_h = self.cfg['RESIZE']
        self.crop_w, self.crop_h = self.cfg['CROP']
        self.compose = albu.Compose
        
        self.mean = self.cfg.MEAN if self.cfg.MEAN is not (0,) else (0.46454108, 0.43718538, 0.39618185)
        self.std = self.cfg.STD if self.cfg.STD is not (0,) else (0.23577851, 0.23005974, 0.23109385)
    
    def get_aug(self, kind):
        if kind == 'val': return self.aug_val()
        elif kind == 'val_forced': return self.aug_val_forced()
        elif kind == 'test': return self.aug_test ()
        elif kind == 'light': return self.aug_light()
        elif kind == 'blank': return self.aug_blank()
        else: raise Exception(f'Unknown aug : {kind}')
        
    def norm(self): return self.compose([albu.Normalize(mean=self.mean, std=self.std), ToTensor()])
    
    def rand_crop(self):
        return albu.OneOf([
                #albu.RandomResizedCrop(h,w, scale=(0.05, 0.4)), 
                albu.RandomCrop(self.crop_h,self.crop_w)
                #albu.CropNonEmptyMaskIfExists(h,w)
            ], p=1)    
    
    def resize(self): return albu.Resize(self.resize_h, self.resize_w)

    def aug_val(self): return self.compose([albu.CenterCrop(self.crop_h,self.crop_w), self.norm()])
    def aug_val_forced(self): return self.compose([albu.CropNonEmptyMaskIfExists(self.crop_h,self.crop_w), self.norm()])
    def aug_test(self): return self.compose([albu.CenterCrop(self.crop_h,self.crop_w), self.norm()])
    def aug_light(self): return self.compose([self.rand_crop(), albu.Flip(), albu.RandomRotate90(), self.norm()])
    def aug_blank(self): return self.compose([self.resize()])


def get_aug(aug_type, transforms_cfg):
    """ aug_type (str): one of `val`, `test`, `light`, `medium`, `hard`
        transforms_cfg (dict): part of main cfg
    """
    auger = Augmentator(cfg=transforms_cfg)
    return auger.get_aug(aug_type)
