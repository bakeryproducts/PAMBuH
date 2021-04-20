import random
from pathlib import Path

import cv2
import numpy as np
from utils import rgb2gray
import albumentations as albu
from PIL import Image


class AddPalingRegion(albu.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, p=.5, alpha=0.7, region_alpha=0.7, thresh_shift=10, alpha_shift=0.05,
                 max_gray_val=255):
        """
        p: probability to apply paling transform
        """
        super(AddPalingRegion, self).__init__(p=p)
        self.alpha = alpha
        self.region_alpha = region_alpha
        self.thresh_shift = thresh_shift
        self.alpha_shift = alpha_shift
        self.max_gray_val = max_gray_val

    @staticmethod
    def _add_rand_shift(number, shift):
        """Adds random shift to number.
        """
        return number + np.random.rand() * 2 * shift - shift

    def _get_thresholds(self, img):
        """Returns thresholds.
        """
        mean, std = img.mean(), img.std()
        thin_border_thresh = AddPalingRegion._add_rand_shift(mean - std, self.thresh_shift)
        full_region_thresh = AddPalingRegion._add_rand_shift(mean, self.thresh_shift)
        return thin_border_thresh, full_region_thresh

    def _get_region_mask(self, img):
        """Returns mask of region shape of (h,w).
            IMG, (h, w, 3)
        """
        thin_border_thresh, full_region_thresh = self._get_thresholds(img)
        gray = rgb2gray(img).astype(np.uint8)

        _, thin_border_mask = cv2.threshold(gray, thin_border_thresh,
                                            self.max_gray_val, cv2.THRESH_BINARY_INV)  # hw
        _, full_region_mask = cv2.threshold(gray, full_region_thresh,
                                            self.max_gray_val, cv2.THRESH_BINARY_INV)  # hw
        return full_region_mask - thin_border_mask  # hw

    def _get_white_img_uint8(self, shape):
        """Returns white img.
        """
        return self.max_gray_val * np.ones(shape).astype(np.uint8)

    def _add_paling(self, img):
        """Modifies/changes img_mean and img_std to img and mean.
            IMG, (h, w, 3)
        """
        assert img.shape[2] == 3  # c
        assert img.shape[0] == img.shape[1]  # hw
        h, w, c = img.shape

        region_mask = self._get_region_mask(img)  # hw
        mask = np.dstack((region_mask, region_mask, region_mask))  # hw3
        mask_bit = (mask / 255).astype(bool)  # hw3 containing either 1 or 0

        alpha = AddPalingRegion._add_rand_shift(self.alpha, self.alpha_shift)
        white_img = self._get_white_img_uint8((h, w, c))  # hw3
        img = cv2.addWeighted(img, alpha, white_img, 1 - alpha, 0.0)  # Pale full image

        region_alpha = AddPalingRegion._add_rand_shift(self.region_alpha, self.alpha_shift)
        blended = cv2.addWeighted(img, region_alpha, mask, 1 - region_alpha, 0.0) # Pale regions
        img[mask_bit] = blended[mask_bit]
        return img.astype(np.uint8)

    def apply(self, img, **params):
        return self._add_paling(img)


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
    def __init__(self, masks_path, alpha=.7, p=.5, base_r=145, base_g=85, base_b=155, shift=30):
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