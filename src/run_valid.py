import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import time
import datetime
from pathlib import Path
from logger import logger
from itertools import cycle
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio as rio

import utils
from run_inference import start_inf
from loss import dice_loss


def get_split_by_idx(idx):
    imgs = Path('input/hm/train').glob('*.tiff')
    splits = [
        ['0486052bb', 'e79de561c'],
        ['2f6ecfcdf', 'afa5e8098'],
        ['1e2425f28', '8242609fa'],
        ['cb2d976f4', 'c68fe75ea'],
        ]
    return [i for i in imgs if i.stem in splits[idx]]

def valid_dice(a,b, eps=1e-6):
    intersection = (a * b).sum()
    dice = ((2. * intersection + eps) / (a.sum() + b.sum() + eps)) 
    return dice

def calc_all_dices(res_masks, thrs):
    dices = {}
    for img_name, mask_name in res_masks.items():
        logger.log('DEBUG', f'Dice for {img_name}')
        gt = rio.open(f'input/masks/bigmasks/{img_name.name}').read()
        mask = np.load(mask_name)
        mask = utils.sigmoid(mask)
        name = img_name.stem

        ds = []
        gt = gt > 0
        for thr in tqdm(thrs):
            th_mask = (mask > thr)
            dice = valid_dice(gt, th_mask)
            ds.append(dice)

        ds = np.array(ds)
        print(name, ds.max(), thrs[np.argmax(ds)])
        dices[name] = ds
    return dices

def calc_common_dice(dices, thrs):
    best_thrs = []
    for k, v in dices.items():
        thr = thrs[np.argmax(v)]
        best_thrs.append(thr)
    return np.mean(best_thrs)

#def get_thrs(): return np.arange(.2,.9, .025)
def get_thrs(): return np.arange(.2,.9, .1)

def get_stats(dices, thrs):
    df = pd.DataFrame(columns=['name', 'thr', 'score', 'real_thr', 'real_score'])

    targ = calc_common_dice(dices, thrs)
    best_idx = np.argmin((thrs - targ)**2)
    best_thr = thrs[best_idx]

    for i, (k, v) in enumerate(dices.items()):
        idx = np.argmax(v)
        df.loc[i] = ([k, thrs[idx], v[idx], best_thr, v[best_idx]])

    s = df.mean()
    s['name'] = 'AVE'
    df.loc[len(df)] = s
    return df


def start_valid(model_folder, split_idx):
    os.makedirs(f'{model_folder}/predicts/masks', exist_ok=True)
    gpu_list = [2,3]
    threshold = 0
    MERGE=False
    num_processes = len(gpu_list)
    save_predicts=True
    use_tta=False
    to_rle=False
    timestamped = False

    img_names = get_split_by_idx(split_idx)
    res_masks = start_inf(model_folder, img_names, gpu_list, num_processes, use_tta, threshold, save_predicts, to_rle)
    logger.log('DEBUG', 'Predicts done')

    #for img_name, mask_name in res_masks.items():
    #    utils.save_tiff_uint8_single_band(v, f'{model_folder}/predicts/cv/masks/{k.name}', bits=8)
    #logger.log('DEBUG', 'Predicts saved')

    thrs = get_thrs()
    dices = calc_all_dices(res_masks, thrs)
    df_result = get_stats(dices, thrs).round(5)
    print(df_result)
    if timestamped:
        timestamp = '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())
        total_name = f'{model_folder}/total_{timestamp}.csv'
    else:
        total_name = f'{model_folder}/total.csv'
    df_result.to_csv(total_name)
    logger.log('DEBUG', f'Total saved {total_name}')

    if MERGE:
        os.makedirs(f'{model_folder}/predicts/combined', exist_ok=True)
        logger.log('DEBUG', 'Merging masks')
        for k, v in res_masks.items():
            mask_name1 = f'input/masks/bigmasks/{k.name}'
            mask_name2 = f'{model_folder}/predicts/masks/{k.name}'
            merge_name = f'{model_folder}/predicts/combined/{k.name}'
            logger.log('DEBUG', f'Merging {merge_name}')
            utils.tiff_merge_mask(k, mask_name1, merge_name, mask_name2)

def join_totals(model_folder):
    totals = list(Path(model_folder).rglob('total.csv'))
    print(totals)
    dfs = [pd.read_csv(str(t), index_col=0).loc[:1] for t in totals] 

    df = pd.concat(dfs)
    s = df.mean()
    s['name'] = 'AVE'
    df.loc[len(df)] = s
    print(df)
    #df.to_csv(f'{model_folder}/total.csv')

if __name__ == '__main__':
    model_folder = Path('output/2021_Apr_13_09_48_24_PAMBUH/')
    FOLDS = [0]#[0,1,2,3] # or int 
    if isinstance(FOLDS, list):
        for split_idx in FOLDS:
            fold_folder = model_folder / f'fold_{split_idx}' 
            start_valid(fold_folder, split_idx)
    else:
        start_valid(model_folder, FOLDS)
    
    join_totals(model_folder)
