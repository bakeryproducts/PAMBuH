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
    
        self.Alights = partial(AddLightning, imgs_path='input/aug_data/light/') # cfg is partial! there is no cfg.INPUTS here
        self.FakeGlo = partial(AddFakeGlom, masks_path='input/CUTS/cuts_B_1536x33/masks/') # cfg is partial! there is no cfg.INPUTS here


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
    def blur(self, p): return albu.OneOf([
                        albu.GaussianBlur((3,9)),
                        #albu.MotionBlur(),
                        #albu.MedianBlur()
                    ], p=p)
    def scale(self, p): return albu.ShiftScaleRotate(0.0625, 0.2, 15, p=p)
    def cutout(self, p): return albu.OneOf([
            albu.Cutout(8, 32, 32, 0,p=.3),
            albu.GridDropout(0.5, fill_value=230, random_offset=True, p=.3),
            albu.CoarseDropout(16, 48, 48, 8, 8, 8, 220, p=.4)
        ],p=p)
    
    def d4(self): return self.compose([
                    albu.Flip(),
                    albu.RandomRotate90()
                    ]) 

    def multi_crop(self): return albu.OneOf([
                    albu.RandomCrop(self.crop_h,self.crop_w),
                    #albu.RandomResizedCrop(self.crop_h, self.crop_w, scale=(0.8, 1)), 
                    #albu.CenterCrop(self.crop_h,self.crop_w, p=.2)
                    ], p=1)    

    def color_jit(self, p): return albu.OneOf([
                    #albu.HueSaturationValue(10,15,10),
                    #albu.CLAHE(clip_limit=4),
                    #albu.RandomBrightnessContrast(.3, .3),
                    albu.ColorJitter(brightness=.4, contrast=0.3, saturation=0.1, hue=0.1)
                    ], p=p)

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
                                                    self.multi_crop(), 
                                                    self.d4(),
                                                    self.additional_res(),
                                                    self.norm()
                                                    ])

    def additional_res(self):
        return self.compose([
                    self.scale(p=.1),
                    self.FakeGlo(p=.1),
                    self.Alights(p=.4),
                    self.color_jit(p=.2),
                    self.cutout(p=.2),
                    self.blur(p=.2),
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
        """
        p: probability to apply blending transform
        d4_prob: probability to apply d4 transform
        """
        super(_AddOverlayBase, self).__init__(p=p)
        self.alpha = alpha
        if np.allclose(self.alpha, 0.0): raise Exception

        self.beta = 1 - alpha
        self.gamma = 0.0
        assert 0 <= self.alpha <= 1, f"Invalid alpha value equal to {self.alpha} (from 0.0 to 1.0)"
        self.get_overlay_fn = get_overlay_fn

    def flip(self, dst, p=.5):
        if random.random() < p:
            #axes = [1, 2, (1, 2)]  # Includes h or w
            axes = [0, 1, (0, 1)]  # Includes h or w
            dst = np.flip(dst, random.choice(axes))
        return dst

    def rotate90(self, dst, p=.5):
        if random.random() < p:
            num_rotates = [1, 2, 3]  # 90, 180 and 270 counterclockwise
            axes = (0, 1)  # Includes hw
            dst = np.rot90(dst, random.choice(num_rotates), axes=axes)
        return dst

    def d4(self, img, p=.5):
        if random.random() < p:
            tr = albu.Compose([albu.Flip(), albu.RandomRotate90()])
            return tr(image=img)['image']

        return img

    def _d4(self, image, p=.5):
        if random.random() < p:
            flip_prob , rotate_prob = .5 , .5
            return self.rotate90(self.flip(image, p=flip_prob), p=rotate_prob)
        return image

    def _blend(self, image1, image2): return cv2.addWeighted(image1, self.alpha, image2, self.beta, self.gamma)

    def alpha_blend(self, image, aug_image):
        """
        IMAGE, (h, w, 3)
        AUG_IMAGE,  (h, w, 4) containing mask in last channel/band
        """
        assert image.shape[2] == 3 and aug_image.shape[2] == 4, (image.shape, aug_image.shape)  # c
        assert image.shape[:2] == aug_image.shape[:2]  , (image.shape, aug_image.shape)  # hw

        aug_image = self.d4(aug_image, p=1)
        rgb, mask = aug_image[...,:3], aug_image[...,3] > 0
        blended = self._blend(rgb, image)
        image[mask] = blended[mask]
        return image

    def apply(self, img, **params):
        return self.alpha_blend(img, self.get_overlay_fn())


class AddLightning(_AddOverlayBase):
    def __init__(self, imgs_path, alpha=1, p=.5):
        super(AddLightning, self).__init__(get_overlay_fn=self.get_lightning, alpha=alpha, p=p)
        self.imgs = list(Path(imgs_path).rglob('*.png'))
        assert len(self.imgs) > 0, imgs_path

    def _expander(self, img): return np.array(img)

    def get_lightning(self):
        img = Image.open(str(random.choice(self.imgs)))
        return self._expander(img)


class AddFakeGlom(_AddOverlayBase):
    def __init__(self, masks_path, alpha=.5, p=.5, base_r=145, base_g=85, base_b=155, shift=30):
        """Default base_r, base_r, base_b and shift are chosen from hist."""
        super(AddFakeGlom, self).__init__(get_overlay_fn=self.get_glom, alpha=alpha, p=p)
        self.masks = list(Path(masks_path).rglob('*.png'))
        self.base_r, self.base_g, self.base_b = base_r, base_g, base_b
        self.shift = shift
        self._shape = 256

    def _expander(self, img): 
        img = np.array(img)
        img = albu.RandomCrop(self._shape, self._shape)(image=img)['image']
        #print(img.shape)
        return img

    def _aug_with_rand_rgb(self, mask):
        """Returns aug_image shape of (h, w, 4) containing rgb and mask:
            - rgb pixel values are integers randomly drawn colour
            - mask pixel values are either 0 or 255
        """
        h, w, c = mask.shape  # hw1
        assert c == 3, f"Invalid number of channels, {c}"
        mask = np.expand_dims(mask[...,0], -1)

        shift_r, shift_g, shift_b = np.random.randint(-self.shift, self.shift, size=3)
        r = np.full((h, w, 1), self.base_r + shift_r, dtype=np.uint8)  # hw1
        g = np.full((h, w, 1), self.base_g + shift_g, dtype=np.uint8)  # hw1
        b = np.full((h, w, 1), self.base_b + shift_b, dtype=np.uint8)  # hw1
        return np.concatenate((r, g, b, mask), axis=2)  # hw4

    def get_glom(self):
        mask = Image.open(str(random.choice(self.masks)))  # hw1
        mask = self._expander(mask)
        return self._aug_with_rand_rgb(mask)  # hw4


def get_aug(aug_type, transforms_cfg, tag=False):
    """ aug_type (str): one of `val`, `test`, `light`, `medium`, `hard`
        transforms_cfg (dict): part of main cfg
    """
    compose = albu.Compose #if not tag else partial(albu.Compose, additional_targets={'mask1':'mask', 'mask2':'mask'})
    auger = Augmentator(cfg=transforms_cfg, compose=compose)
    return auger.get_aug(aug_type)

