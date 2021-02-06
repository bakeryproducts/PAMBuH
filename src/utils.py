import json
import os
from pathlib import Path
from functools import partial
from typing import Tuple, List, Dict, Callable

import cv2
from shapely import geometry
import numpy as np


def jread(path: str) -> Dict:
    with open(str(path), 'r') as f:
        data = json.load(f)
    return data


def jdump(data: Dict, path: str) -> None:
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=4)


def filter_ban_str_in_name(s: str, bans: List[str]): return any([(b in str(s)) for b in bans])


def get_filenames(path: str, pattern: str, filter_out_func: Callable) -> str:
    """
    pattern : "*.json"
    filter_out : function that return True if file name is acceptable
    """

    filenames = list(Path(path).glob(pattern))
    assert (filenames), f'There is no matching filenames for {path}, {pattern}'
    filenames = [fn for fn in filenames if not filter_out_func(fn)]
    assert (filenames), f'There is no matching filenames for {filter_out_func}'
    return filenames


def polyg_to_mask(polyg: np.ndarray, wh: Tuple[int, int], fill_value: int) -> np.ndarray:
    """Convert polygon to binary mask.
    """

    polyg = np.int32([polyg])
    mask = np.zeros([wh[0], wh[1]], dtype=np.uint8)
    cv2.fillPoly(mask, polyg, fill_value)
    return mask


def json_record_to_poly(record: List) -> List:
    coords = record['geometry']['coordinates'][0] # 0 as in first part of multipart poly
    try: poly = geometry.Polygon(coords)
    except Exception as e: print(e, coords)
    return poly


def create_dir(path: str) -> None:
    """ Create directory if ot existed.
    """

    if not os.path.isdir(path):
        os.makedirs(path)



