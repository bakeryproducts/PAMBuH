import json
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon
import rasterio
from rasterio.windows import Window

from utils import jread


class GdalSampler:
    """Iterates over img with annotation, returns tuples of img, mask
    """

    def __init__(self, img_path: str, mask_path: str, img_polygs_path: str, wh: Tuple[int], mask_wh: Tuple[int]):
        self._markups = jread(img_polygs_path)
        self._masks = rasterio.open(mask_path)
        self._img = rasterio.open(img_path)

        self._wh = wh
        self._mask_wh = mask_wh
        self._bands_img = (1, 2, 3)
        self._bands_mask = 1
        self._count = -1
        # Get polygons
        self.polygs = [polyg['geometry']['coordinates'][0] for polyg in self._markups]
        self.polygs_norm = [(np.array(polyg) - np.array(Polygon(polyg).centroid) + np.array([mask_wh[0] // 2,
                                                                                             mask_wh[0] // 2])).
                                tolist() for polyg in self.polygs]
        polygs_cent_coord = [np.round(Polygon(polyg).centroid) for polyg in self.polygs]
        # Get offsets
        self.xoff_img = [int(polyg_cent_coord[0] - self._wh[0] // 2) for polyg_cent_coord in polygs_cent_coord]
        self.yoff_img = [int(polyg_cent_coord[1] - self._wh[1] // 2) for polyg_cent_coord in polygs_cent_coord]
        self.xoff_mask = [int(polyg_cent_coord[0] - self._mask_wh[0] // 2) for polyg_cent_coord in polygs_cent_coord]
        self.yoff_mask = [int(polyg_cent_coord[1] - self._mask_wh[1] // 2) for polyg_cent_coord in polygs_cent_coord]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._markups)

    def __next__(self):
        self._count += 1
        if self._count < len(self._markups):
            return self.__getitem__(self._count)
        else:
            self._count = -1
            raise StopIteration("Failed to proceed to the next step")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        window_img = Window(self.xoff_img[idx], self.yoff_img[idx], *self._wh)
        window_mask = Window(self.xoff_mask[idx], self.yoff_mask[idx], *self._mask_wh)
        img = self._img.read(self._bands_img, window=window_img)
        mask = self._masks.read(self._bands_mask, window=window_mask)
        return img, mask
