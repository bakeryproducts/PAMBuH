import numpy as np
import cv2
from typing import Tuple


def polyg_to_mask(polyg: np.ndarray, wh: Tuple[int, int], fill_value: int) -> np.ndarray:
    """Convert polygon to mask where 1 - polygon and 0 - no polygon.
    """

    polyg = np.int32([polyg])
    mask = np.zeros([wh[0], wh[1]], dtype=np.uint8)
    cv2.fillPoly(mask, polyg, fill_value)
    return mask
