import numpy as np
import cv2
from typing import Tuple

import utils


def cut_off_light_from_img(img: np.ndarray,
                           threshold: int = 210,
                           kernel_size: int = 5,
                           min_light_ratio: float = 0.005,
                           max_light_ratio: float = 0.5) -> Tuple[None, np.ndarray]:
    """Cuts off lightning from img.

    ratio: Defines the ratio of kernel size compared to img.
           Smaller value, thinner lightnings (varies from 0.0 to 1.0)
    min_light_cell_ratio: Min possible ratio (total number of white cells / total number of cells)
    max_light_ratio: Min possible ratio (total number of white cells / total number of cells)

    Attention: Img may contain either 1 or 3 channel(s)

    """

    max_gray_val = 255
    num_channels, w, h = img.shape

    assert num_channels in [1, 3], "Invalid number of channels"
    if num_channels == 3:
        img = np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)  # shift axis by one left
        img = utils.rgb2gray(img)

    kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    _, thresh = cv2.threshold(img, threshold, max_gray_val, cv2.THRESH_BINARY)

    total_num_cells = w * h
    num_white_cells = thresh.sum() / max_gray_val
    ratio = num_white_cells / total_num_cells

    if min_light_ratio <= ratio <= max_light_ratio:
        return thresh[0]
