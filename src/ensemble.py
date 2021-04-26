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


def ensemble_predictions(folders, dst_path, logits):
    mask_names = (folders[0]/'predicts/raw').glob('*.npy')
    mask_names = [m.name for m in mask_names]
    for mask_name in tqdm(mask_names):
        logger.log("DEBUG", f'{mask_name}') 

        ens_mask = average_preds(mask_name, folders, logits=logits)
        mask_path = dst_path / 'predicts/raw' 
        os.makedirs(mask_path, exist_ok=True)
        np.save(str(mask_path / mask_name), ens_mask)

def average_preds(name, folders, logits=True):
    masks = []
    for folder in folders:
        mask = np.load(str(folder / 'predicts/raw' / name))
        logger.log("DEBUG", f'{name}') 
        if not logits: mask = utils.sigmoid(mask)
        masks.append(mask)
    ens_mask = np.mean(masks,0)
    return ens_mask


def ensemble_folds(folders, dst_path, logits):
    for i in range(4):
        fold_folders = [p / f'fold_{i}' for p in folders]
        dst_folder_path = dst_path / f'fold_{i}'
        os.makedirs(dst_folder_path, exist_ok=True)
        ensemble_predictions(fold_folders, dst_folder_path, logits)

    

if __name__ == '__main__':
    FOLDS = True
    LOGITS = False
    folders = [
            Path('output/2021_Apr_24_12_05_27_PAMBUH/'),
            Path('output/2021_Apr_24_00_57_16_PAMBUH/'),
            Path('output/2021_Apr_14_21_08_06_PAMBUH/'),
            Path('output/2021_Apr_26_08_32_37_PAMBUH/'),
            ]
    timestamp = '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())
    dst_path = Path(f'output/{timestamp}')
    os.makedirs(dst_path, exist_ok=True)

    if FOLDS: ensemble_folds(folders, dst_path, LOGITS)
    else: ensemble_predictions(folders, dst_path, LOGITS)

