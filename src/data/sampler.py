import numpy as np
import json
from shapely.geometry import Polygon
from typing import List, Tuple
# from osgeo import gdal
import rasterio
from rasterio.windows import Window

from src.utilis import polyg_to_mask


class Sampler(object):
    """Sample tiff file/img and return sampled_img and mask where 1 - glomerulus and 0 - no glomerulus.
    """

    def __init__(self, img_path: str, img_markup_path: str, wh: Tuple[int], mask_wh: Tuple[int]):
        with open(img_markup_path) as json_file:
            self._markups = json.load(json_file)
        # self._img = gdal.Open(img_path, gdal.GA_ReadOnly)
        self._img = rasterio.open(img_path)
        self._wh = wh
        self._mask_wh = mask_wh

        self.count = -1

    def read_img(self, key: int):
        """ Read/slice image.
        """
        # List of coordinates
        polyg = self._markups[key]['geometry']['coordinates'][0]
        # (x_cent,y_cent)
        polyg_cent_coord = np.round(Polygon(polyg).centroid)
        # Polygon with center in (0, 0)
        polyg_norm = np.array(polyg) - np.mean(polyg, axis=0).tolist()
        # Set window params
        xoff, yoff = int(polyg_cent_coord[0] - self._wh[0] // 2), int(polyg_cent_coord[1] - self._wh[1] // 2)
        xsize, ysize = self._wh[0], self._wh[1]
        return self._img.read((1, 2, 3), window=Window(xoff, yoff, xsize, ysize)), \
               polyg_to_mask(polyg_norm, self._mask_wh)

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        if self.count < len(self._markups):
            return self.read_img(self.count)
        else:
            self.count = -1
            raise StopIteration("Failed to proceed to the next step")

    def __len__(self):
        return len(self._markups)

    def __getitem__(self, key: int) -> Tuple[np.ndarray, np.ndarray]:
        if key < len(self._markups):
            return self.read_img(key)
        else:
            raise IndexError
