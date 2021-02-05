from typing import Tuple

import cv2
import numpy as np


def polyg_to_mask(polyg: np.ndarray, wh: Tuple[int, int], fill_value: int) -> np.ndarray:
    """Convert polygon to binary mask
    """
    polyg = np.int32([polyg])
    mask = np.zeros([wh[0], wh[1]], dtype=np.uint8)
    cv2.fillPoly(mask, polyg, fill_value)
    return mask
