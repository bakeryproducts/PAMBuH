import os
import time
from pathlib import Path
from logger import logger
from itertools import cycle
from functools import partial
import multiprocessing as mp
import multiprocessing.pool

import numpy as np
from tqdm import tqdm

from postp import start, dump_to_csv, mp_func_wrapper
import utils
import rle2tiff



def start_inf(model_folder, img_names, gpu_list, num_processes, use_tta, threshold=.5, save_predicts=False, to_rle=False):
    logger.log('DEBUG', '\n'.join(list([str(i) for i in img_names])))

    m = mp.Manager()
    results = m.dict()
    gpus = m.dict()
    for i in gpu_list:
        gpus[i] = False

    starter = partial(start, model_folder=model_folder, gpus=gpus, results=results, use_tta=use_tta,)
    starter = partial(mp_func_wrapper, starter)
    args = [(name,) for name in img_names]

    with utils.NoDaemonPool(num_processes) as p:    
        g = tqdm(p.imap(starter, args), total=len(img_names))
        for _ in g:
            pass

    result_masks = dict(results)

    for img_name, mask_path in result_masks.items():
        mask = np.load(mask_path)[0]
        if save_predicts:
            mask = utils.sigmoid(mask)
            #mask = (mask > threshold).astype(np.uint8)

            out_name = model_folder/'predicts/masks'/img_name.name
            os.makedirs(str(out_name.parent), exist_ok=True)
            utils.save_tiff_uint8_single_band((255 * mask).astype(np.uint8), str(out_name), bits=8)

            logger.log('DEBUG', f'{img_name} done')
            if to_rle:
                logger.log('DEBUG', f'RLE...')
                mask = (mask > threshold).astype(np.uint8)
                rle = rle2tiff.mask2rle(mask)
                result_masks[img_name] = rle

    return result_masks 



if __name__ == '__main__':
    model_folder = Path('output/2021_Apr_12_19_28_51_PAMBUH/')
    gpu_list = [0,1,2,3]
    threshold = .5
    num_processes = len(gpu_list)

    img_names = list(Path('input/hm/test').glob('*.tiff'))
    img_names = [img_names[i] for i in [2,0,4,1,3]]#aa first
    #img_names = img_names[-2:]

    use_tta = True
    save_predicts = True
    to_rle = True

    results = start_inf(model_folder, 
                        img_names, 
                        gpu_list, 
                        num_processes, 
                        use_tta, 
                        threshold=threshold, save_predicts=save_predicts, to_rle=to_rle)

    if to_rle: dump_to_csv(results, model_folder, threshold)
