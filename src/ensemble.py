import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import time
import datetime
from pathlib import Path
from logger import logger
from itertools import cycle
from functools import partial

import numpy as np
from tqdm import tqdm

import utils


def ensemble_predictions(folders, dst_path):
    save_masks=False
    mask_names = (folders[0]/'predicts/raw').glob('*.npy')
    #mask_names = [m.name for m in mask_names]
    for mask_full in tqdm(mask_names):
        mask_name = mask_full.name
        logger.log("DEBUG", f'{mask_name}') 

        ens_mask = average_preds(mask_name, folders)
        mask_path = dst_path / 'predicts/raw' 
        os.makedirs(mask_path, exist_ok=True)
        np.save(str(mask_path / mask_name), ens_mask)

        if save_masks:
            logger.log("DEBUG", f'MAX:{ens_mask.max(), ens_mask.dtype, ens_mask.shape}') 
            mask_path = dst_path / 'predicts/masks' 
            os.makedirs(mask_path, exist_ok=True)
            utils.save_tiff_uint8_single_band((255 * ens_mask.squeeze()).astype(np.uint8), str(mask_path/mask_full.stem) + '.tiff', bits=8)

def average_preds(name, folders):
    masks = []
    for folder in folders:
        mask = np.load(str(folder / 'predicts/raw' / name))
        logger.log("DEBUG", f'{name}') 
        masks.append(mask)
    ens_mask = np.mean(masks,0)
    return ens_mask


def ensemble_folds(folders, dst_path):
    for i in range(4):
        fold_folders = [p / f'fold_{i}' for p in folders]
        dst_folder_path = dst_path / f'fold_{i}'
        os.makedirs(dst_folder_path, exist_ok=True)
        ensemble_predictions(fold_folders, dst_folder_path)

    
if __name__ == '__main__':
    FOLDS = True
    folders = [
            Path('output/S/2021_May_05_01_18_57_PAMBUH_s_L/'),
            Path('output/S/2021_May_07_11_36_31_PAMBUH_s/'),
            Path('output/S/2021_May_07_14_59_23_PAMBUH_s/'),
            Path('output/S/2021_May_07_18_01_35_PAMBUH_s/'),
            ]
    timestamp = '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())
    dst_path = Path(f'output/{timestamp}')
    os.makedirs(dst_path, exist_ok=True)

    if FOLDS: ensemble_folds(folders, dst_path)
    else: ensemble_predictions(folders, dst_path)

