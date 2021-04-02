from typing import Tuple
import numpy as np
import rasterio
from rasterio.windows import Window


class RasterioReader:
    """Reads tiff files.

    If num of bands is equal to 1, then treats subdatasets as bands.
    """

    def __init__(self, path_to_tiff_file: str, num_out_bands: Tuple[int, None] = None):
        self.ds = rasterio.open(path_to_tiff_file)
        self.num_bands = self.ds.count
        self.num_out_bands = num_out_bands if num_out_bands is not None else self.num_bands
        assert self.num_bands in [1, self.num_out_bands], \
                    f'Invalid number of input bands equal to {self.num_bands}'

        if self.num_bands != self.num_out_bands:
            path_to_subdatasets = self.ds.subdatasets
            self.list_ds = [rasterio.open(path_to_subdataset)
                            for path_to_subdataset in path_to_subdatasets]
            assert len(self.list_ds) == num_out_bands, \
                f'Invalid number of input subdatasets equal to {len(self.list_ds)}'

    def read(self, window: Tuple[None, Window] = None, boundless: bool=True):

        if self.num_bands != self.num_out_bands:
            output = np.vstack([ds.read() for ds in self.list_ds]) if window is None else \
                np.vstack([ds.read(window=window, boundless=boundless) for ds in self.list_ds])
        else:
            output = self.ds.read() if window is None else \
                self.ds.read(window=window, boundless=boundless)
        return output

    def __del__(self):
        del self.ds
        if self.num_bands != self.num_out_bands:
            del self.list_ds
