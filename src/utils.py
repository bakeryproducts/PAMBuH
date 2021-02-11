import os
import json
import argparse
import datetime
from pathlib import Path
from functools import partial
import multiprocessing as mp
from contextlib import contextmanager
from typing import Tuple, List, Dict, Callable
from rasterio.windows import Window

import cv2
import torch
import rasterio
import numpy as np
from shapely import geometry

from config import cfg


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


def json_record_to_poly(record: Dict) -> List[geometry.Polygon]:
    """Get list of polygs from record.
    """

    num_polygs = len(record['geometry']['coordinates'])
    if num_polygs == 1:     # Polygon
        list_coords = [record['geometry']['coordinates'][0]]
    elif num_polygs > 1:    # MultiPolygon
        list_coords = [record['geometry']['coordinates'][i][0] for i in range(num_polygs)]
    else:
        raise Exception("No polygons are found")

    try:
        polygs = [geometry.Polygon(coords) for coords in list_coords]
    except Exception as e:
        print(e, list_coords)
    return polygs


def make_folders(cfg):
    cfg_postfix = 'PAMBUH'
    timestamp = '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())
    name = Path(timestamp + '_' + cfg_postfix)
    fname = cfg.OUTPUTS / name
    os.makedirs(fname, exist_ok=True)
    img_dir = fname / 'imgs'
    os.makedirs(img_dir, exist_ok=True)
    models_dir = fname / 'models'
    os.makedirs(models_dir, exist_ok=True)
    tb_dir = fname / 'tb'
    os.makedirs(tb_dir, exist_ok=True)

    return fname

def save_models(d, postfix, output_folder):
    for k, v in d.items():
        torch.save(v.state_dict(), output_folder/os.path.join("models",f"{k}_{postfix}.pkl")) 

def dump_params(cfg, output_path):
    with open(os.path.join(output_path, 'cfg.yaml'), 'w') as f:
        f.write(cfg.dump())
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    return args

def get_basics_rasterio(name):
    file = rasterio.open(str(name))
    return file, file.shape, file.count

def read_frame(fd, x, y, h, w=None, c=3):
    if w is None: w = h
    img = fd.read(list(range(1, c+1)), window=Window(x, y, h, w))
    return torch.ByteTensor(img)

def save_tiff_uint8_single_band(img, path):
    if isinstance(img, torch.Tensor):
        img = np.array(img)
    elif not isinstance(img, numpy.ndarray):
        raise TypeError(f'Want torch.Tensor or numpy.ndarray, but got {type(img)}')
    assert img.dtype == np.uint8
    h, w = img.shape
    dst = rasterio.open(path, 'w', driver='GTiff', height=h, width=w, count=1, nbits=1, dtype=np.uint8)
    dst.write(img, 1) # 1 band
    dst.close()
    print(f'Save to {path}')
    del dst

def cfg_frz(func):
    def frz(*args, **kwargs):
        cfg.defrost()
        r = func(*args, **kwargs)
        cfg.freeze()
        return r 
    return frz

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def mp_func(foo, args, n):
    args_chunks = [args[i:i + n] for i in range(0, len(args), n)]
    with poolcontext(processes=n) as pool:
        res = pool.map(foo, args_chunks)
    return [ri for r in res for ri in r]


def mp_func_gen(foo, args, n, progress=None):
    args_chunks = [args[i:i + n] for i in range(0, len(args), n)]
    results = []
    with poolcontext(processes=n) as pool:
        gen = pool.imap(foo, args_chunks)
        if progress is not None: gen = progress(gen, total=len(args_chunks))
        for r in gen:
            results.extend(r)
    return results


def get_cortex_polygs(anot_structs_json: Dict) -> List[geometry.Polygon]:
    """ Get list of cortex polygons from anot_structs_json.
    """

    cortex_polygs = []
    for record in anot_structs_json:
        if record['properties']['classification']['name'] == 'Cortex':
            cortex_polygs += json_record_to_poly(record)
    return cortex_polygs
