import json
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon
import rasterio
from rasterio.windows import Window

from utils import polyg_to_mask


class GdalSampler:
    """Sample tiff file/img and return sampled_img and mask where 1 - glomerulus and 0 - no glomerulus.
    """

    def __init__(self, img_path: str, img_markup_path: str, wh: Tuple[int], mask_wh: Tuple[int]):
        with open(img_markup_path) as json_file:
            self._markups = json.load(json_file)
        self._img = rasterio.open(img_path)
        self._wh = wh
        self._mask_wh = mask_wh
        self._bands = (1, 2, 3)
        self._count = -1
        self._fill_value = 1

        self.polygs = [markup['geometry']['coordinates'][0] for markup in self._markups]
        self.polygs_norm = [(np.array(polyg) - np.array(Polygon(polyg).centroid) + np.array([mask_wh[0] // 2, mask_wh[0] // 2])).
                                tolist() for polyg in self.polygs]
        self.polygs_cent_coord = [np.round(Polygon(polyg).centroid) for polyg in self.polygs]
        self.xoff = [int(polyg_cent_coord[0] - self._wh[0] // 2) for polyg_cent_coord in self.polygs_cent_coord]
        self.yoff = [int(polyg_cent_coord[1] - self._wh[1] // 2) for polyg_cent_coord in self.polygs_cent_coord]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._markups)

    def __next__(self):
        self.count += 1
        if self.count < len(self._markups):
            return self.__getitem__(self.count)
        else:
            self._count = -1
            raise StopIteration("Failed to proceed to the next step")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        window = Window(self.xoff[idx], self.yoff[idx], *self._wh)
        img = self._img.read(self._bands, window=window)
        mask = polyg_to_mask(self.polygs_norm[idx], self._mask_wh, self._fill_value)
        return img, mask
