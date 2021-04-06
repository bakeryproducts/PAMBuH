import random
from pathlib import Path
from functools import partial

import cv2
import torch  # to tensor transform
import numpy as np
from PIL import Image
import albumentations as albu
import albumentations.pytorch as albu_pt


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
    
        #self.Lightniner = partial(AddLightning, self.cfg.DATA.AUG.LIGHT_PATH, alpha=1)


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
            albu.Cutout(8, 32, 32, 0,p=.3),
            albu.GridDropout(0.5, fill_value=230, random_offset=True, p=.3),
            albu.CoarseDropout(16, 32, 32, 8, 8, 8, 220, p=.4)
        ],p=.3)
    
    
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
                    albu.ColorJitter(brightness=.3, contrast=0.3, saturation=0.3, hue=0.3)
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
                                                    self.additional_res(),
                                                    self.norm()
                                                    ])

    def additional_res(self):
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

    def share_some_light(self): return self.Lightniner()


class _AddOverlayBase(albu.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, get_overlay_fn, alpha=1, p=.5):
        super(_AddOverlayBase, self).__init__(p=p)
        self.alpha = alpha
        self.beta = 1 - alpha
        self.gamma = 0.0
        assert 0 <= self.alpha <= 1, f"Invalid alpha value equal to {self.alpha} (from 0.0 to 1.0)"
        self.get_overlay_fn = get_overlay_fn
        self._default_prob = .75 # ? why

    def flip(self, dst, p=.5):
        if random.random() < p:
            axes = [1, 2, (1, 2)]  # Includes h or w
            dst = np.flip(dst, random.choice(axes))
        return dst

    def rotate90(self, dst, p=.5):
        if random.random() < p:
            num_rotates = [1, 2, 3]  # 90, 180 and 270 counterclockwise
            axes = (1, 2)  # Includes hw
            dst = np.rot90(dst, random.choice(num_rotates), axes=axes)
        return dst

    def d4(self, dst, p=1): return self.rotate90(self.flip(dst))

    def alpha_blend(self, src, dst):
        """ 
        src: MASK, (4, h, w) containing mask in last channel/band
        dst: IMAGE, (3, h, w)
        """

        assert dst.shape[0] == 3 and src.shape[0] == 4  # c
        assert dst.shape[1:] == src.shape[1:]  # hw

        if self.alpha < np.finfo(float).eps: raise Exception
        elif self.alpha == 1.0: # > 1 -np.finfo(float).eps ?
            src = self.d4(src)
            rgb, mask = src[:3], np.where(src[3] > 0, True, False).astype(int)  # 3hw, hw
            # easy mode TODO: check if it works
            dst[mask] = rgb[mask]
            return dst
        else:
            src = self.d4(src)
            rgb, mask = src[:3], np.where(src[3] > 0, True, False).astype(int)  # 3hw, hw
            blended = cv2.addWeighted(rgb, self.alpha, dst, self.beta, self.gamma)
            blended_cut_by_mask = np.multiply(blended, mask).astype(np.uint8)
            return np.where(blended_cut_by_mask > 0, blended_cut_by_mask, dst)  # 3hw

    def apply(self, img, **params): return self.alpha_blend(self.get_overlay_fn(), img)


class AddLightning(_AddOverlayBase):
    def __init__(self, imgs_path, alpha=1, p=.5):
        super(AddLightning, self).__init__(get_overlay_fn=self.get_lightning, alpha=alpha, p=p)
        self.imgs = list(Path(imgs_path).glob('*.png'))

    def get_lightning(self):
        img = Image.open(str(random.choice(self.imgs)))
        return np.array(img).transpose([2, 0, 1])  # 4, h, w

class AddNoisyPattern(_AddOverlayBase):
    pass


def get_aug(aug_type, transforms_cfg, border=False):
    """ aug_type (str): one of `val`, `test`, `light`, `medium`, `hard`
        transforms_cfg (dict): part of main cfg
    """
    compose = albu.Compose if not border else partial(albu.Compose, additional_targets={'mask1':'mask', 'mask2':'mask'})
    auger = Augmentator(cfg=transforms_cfg, compose=compose)
    return auger.get_aug(aug_type)

