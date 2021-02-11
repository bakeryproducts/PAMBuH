import json
from PIL import Image
from typing import List, Tuple
import random 
import numpy as np
import rasterio
from rasterio.windows import Window

from utils import jread, get_basics_rasterio, json_record_to_poly, flatten_2dlist


class GdalSampler:
    """Iterates over img with annotation, returns tuples of img, mask
    """

    def __init__(self, img_path: str,
                 mask_path: str,
                 img_polygons_path: str,
                 img_wh: Tuple[int, int],
                 mask_wh: Tuple[int, int],
                 rand_shift_range: Tuple[int, int] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        """If rand_shift_range ~ (0,0), then centroid of glomerulus corresponds centroid of output sample
        """
        assert img_wh == mask_wh, "Image and mask wh should be identical"
        self._records_json = jread(img_polygons_path)
        self._mask = rasterio.open(mask_path)
        self._img = rasterio.open(img_path)
        self._img_wh = img_wh
        self._mask_wh = mask_wh
        self._bands_img = (1, 2, 3)
        self._bands_mask = 1
        self._count = -1
        self._rand_shift_range = rand_shift_range
        # Get 1d list of polygons
        polygons = flatten_2dlist([json_record_to_poly(record) for record in self._records_json])
        self._polygons_centroid = [np.round(polygon.centroid) for polygon in polygons]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._records_json)

    def __next__(self):
        self._count += 1
        if self._count < len(self._records_json):
            return self.__getitem__(self._count)
        else:
            self._count = -1
            raise StopIteration("Failed to proceed to the next step")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x_rand_shift, y_rand_shift = random.randint(-self._rand_shift_range[0], self._rand_shift_range[0]), \
                                     random.randint(-self._rand_shift_range[1], self._rand_shift_range[1])
        x_shift, y_shift = -self._img_wh[0]//2 + x_rand_shift, -self._img_wh[1]//2 + x_rand_shift

        window_img = Window(self._polygons_centroid[idx][0]+x_shift,
                            self._polygons_centroid[idx][1]+y_shift, * self._img_wh)
        window_mask = Window(self._polygons_centroid[idx][0]+x_shift,
                             self._polygons_centroid[idx][1]+y_shift, * self._mask_wh)
        img = self._img.read(self._bands_img, window=window_img)
        mask = self._mask.read(self._bands_mask, window=window_mask)
        return img, mask

    def __del__(self):
        del self._records_json
        del self._mask
        del self._img


def _write_block(block, name):
    x, y, block_data = block
    #print(name, x,y,block_data.shape, block_data.dtype)
    t = Image.fromarray(block_data.transpose((1,2,0)))
    t.save(f'output/{name}_{x}_{y}.png')

def tif_block_read(name, block_size=None):
    if block_size is None: block_size = (256, 256)
    input_file, (W,H), _ = get_basics_rasterio(name)

    nXBlocks, nYBlocks = _count_blocks(name, block_size=block_size)
    nXValid, nYValid = block_size[0], block_size[1]
    
    for X in range(nXBlocks):
        if X == nXBlocks - 1: nXValid = W - X * block_size[0]
        myX = X * block_size[0]
        nYValid = block_size[1]
        for Y in range(nYBlocks):
            if Y == nYBlocks - 1: nYValid = H - Y * block_size[1]
            myY = Y * block_size[1]
            
            window = Window(myY, myX, nYValid, nXValid)
            block = input_file.read([1,2,3], window=window)
            #print(myX, myY, nXValid, nYValid, W, H, block.shape)

            yield X, Y, block
    del input_file


def _count_blocks(name, block_size=(256, 256)):
    # find total x and y blocks to be read
    _, dims, *_  = get_basics_rasterio(name)
    nXBlocks = (int)((dims[0] + block_size[0] - 1) / block_size[0])
    nYBlocks = (int)((dims[1] + block_size[1] - 1) / block_size[1])
    return nXBlocks, nYBlocks


