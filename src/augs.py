from functools import partial

import albumentations as albu
import albumentations.pytorch as albu_pt
import torch  # to tensor transform
import numpy as np


class ToTensor(albu_pt.ToTensorV2):
    def apply_to_mask(self, mask, **params): return torch.from_numpy(mask).permute((2,0,1))
    def apply(self, image, **params): return torch.from_numpy(image).permute((2,0,1))

class Augmentator:
    def __init__(self, cfg, compose):
        self.cfg = cfg 
        self.resize_w, self.resize_h = self.cfg['RESIZE']
        self.crop_w, self.crop_h = self.cfg['CROP']
        self.compose = compose
        
        self.mean = self.cfg.MEAN if self.cfg.MEAN is not (0,) else (0.46454108, 0.43718538, 0.39618185)
        self.std = self.cfg.STD if self.cfg.STD is not (0,) else (0.23577851, 0.23005974, 0.23109385)
    
    def get_aug(self, kind):
        if kind == 'val': return self.aug_val()
        elif kind == 'val_forced': return self.aug_val_forced()
        elif kind == 'test': return self.aug_test ()
        elif kind == 'light': return self.aug_light()
        elif kind == 'light_scale': return self.aug_light_scale()
        elif kind == 'blank': return self.aug_blank()
        elif kind == 'resize': return self.resize()
        elif kind == 'wocrop': return self.aug_wocrop()
        elif kind == 'ssl': return self.aug_ssl()
        else: raise Exception(f'Unknown aug : {kind}')
        
    def norm(self): return self.compose([albu.Normalize(mean=self.mean, std=self.std), ToTensor()])
    def rand_crop(self): albu.RandomCrop(self.crop_h,self.crop_w)
    def crop_scale(self): return albu.RandomResizedCrop(self.crop_h, self.crop_w, scale=(.3,.7))
    def resize(self): return albu.Resize(self.resize_h, self.resize_w)
    def blur(self): return albu.OneOf([
                        albu.GaussianBlur((3,9)),
                        #albu.MotionBlur(),
                        #albu.MedianBlur()
                    ], p=.3)
    def scale(self): return albu.ShiftScaleRotate(0.0625, 0.1, 15, p=.2)
    def rotate(self): return albu.Rotate(45)
    def cutout(self): return albu.OneOf([
            albu.Cutout(8, 32, 32, 230,p=.2),
            albu.GridDropout(0.5, fill_value=230, random_offset=True, p=.2),
            albu.CoarseDropout(16, 32, 32, 8, 8, 8, 220, p=.6)
        ],p=.2)
    
    
    def d4(self): return self.compose([
                    albu.Flip(),
                    albu.RandomRotate90()
                    ]) 

    def multi_crop(self): return albu.OneOf([
                    albu.RandomCrop(self.crop_h,self.crop_w),
                    #albu.RandomResizedCrop(self.crop_h, self.crop_w, scale=(0.8, 1)), 
                    #albu.CenterCrop(self.crop_h,self.crop_w, p=.2)
                    ], p=1)    
    def color_jit(self): return albu.OneOf([
                    #albu.HueSaturationValue(10,15,10),
                    albu.CLAHE(clip_limit=4),
                    #albu.RandomBrightnessContrast(.3, .3),
                    albu.ColorJitter(brightness=.2, contrast=0.1, saturation=0.1, hue=0.1)
                    ], p=0.3)

    def aug_ssl(self): return self.compose([
                    albu.OneOf([
                        albu.HueSaturationValue(30,40,30),
                        albu.CLAHE(clip_limit=4),
                        albu.RandomBrightnessContrast((-0.5, .5), .3),
                        albu.ColorJitter(brightness=.5, contrast=0.5, saturation=0.3, hue=0.3)
                    ], p=1),
                    self.cutout(),
                    self.blur(),
                    self.norm()
                    ])

    def aug_light_scale(self): return self.compose([
                                                    #self.scale(),
                                                    self.multi_crop(), 
                                                    self.d4(),
                                                    self.aug_res(),
                                                    self.norm()
                                                    ])

    def aug_res(self):
        return self.compose([
                    self.color_jit(),
                    self.cutout(),
                    self.blur(),
            ], p=.5)

    def aug_val_forced(self): return self.compose([albu.CropNonEmptyMaskIfExists(self.crop_h,self.crop_w), self.norm()])
    def aug_val(self): return self.compose([albu.CenterCrop(self.crop_h,self.crop_w), self.norm()])
    def aug_light(self): return self.compose([albu.CenterCrop(self.crop_h,self.crop_w, p=1), albu.Flip(), albu.RandomRotate90(), self.norm()])
    def aug_wocrop(self): return self.compose([self.resize(), albu.Flip(), albu.RandomRotate90(), self.norm()])
    def aug_blank(self): return self.compose([self.resize()])
    def aug_test(self): return self.compose([self.resize(), self.norm()])

def get_aug(aug_type, transforms_cfg, border=False):
    """ aug_type (str): one of `val`, `test`, `light`, `medium`, `hard`
        transforms_cfg (dict): part of main cfg
    """
    compose = albu.Compose if not border else partial(albu.Compose, additional_targets={'mask1':'mask', 'mask2':'mask'})
    auger = Augmentator(cfg=transforms_cfg, compose=compose)
    return auger.get_aug(aug_type)

