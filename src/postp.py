import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import json
import time
import random
from pathlib import Path
import multiprocessing as mp
from functools import partial
from logger import logger

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio as rio

import utils
import rle2tiff

import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def _count_blocks(dims, block_size):
    nXBlocks = (int)((dims[0] + block_size[0] - 1) / block_size[0])
    nYBlocks = (int)((dims[1] + block_size[1] - 1) / block_size[1])
    return nXBlocks, nYBlocks

def get_basics_rasterio(name):
    file = rio.open(str(name))
    return file, file.shape, file.count

def generate_block_coords(H, W, block_size):
    h,w = block_size
    nYBlocks = (int)((H + h - 1) / h)
    nXBlocks = (int)((W + w - 1) / w)
    
    for X in range(nXBlocks):
        cx = X * h
        for Y in range(nYBlocks):
            cy = Y * w
            yield cy, cx, h, w

            
def pad_block(y,x,h,w, pad): return np.array([y-pad, x-pad, h+2*pad, w+2*pad])
def crop(src, y,x,h,w): return src[..., y:y+h, x:x+w]
def crop_rio(p, y,x,h,w):
    ds = rio.open(p)
    block = ds.read(window=((y,y+h),(x,x+w)), boundless=True)
    ds.close()
    del ds
    return block

def paste(src, block, y,x,h,w):src[..., y:y+h, x:x+w] = block
def paste_crop(src, part, block_cd, pad):
    _,H,W = src.shape
    y,x,h,w = block_cd
    h, w = min(h, H-y), min(w, W-x)  
    part = crop(part, pad, pad, h, w)
    paste(src, part, *block_cd)

def infer_blocks(blocks, do_inference):
    blocks = do_inference(blocks).cpu().numpy()    
    return (255*blocks).astype(np.uint8)
 
def image_q_reader(q, p, cds, pad):
    for block_cd in cds:
        padded_block_cd = pad_block(*block_cd, pad)
        block = crop_rio(p, *(padded_block_cd))
        #if block.shape[0] == 1: block = np.repeat(block, 3, 0)
        q.put((block, block_cd))
        
def mp_func_wrapper(func, args): return func(*args)
def chunkify(l, n): return [l[i:i + n] for i in range(0, len(l), n)]

def filter_mask(mask, name):
    with open(str(name.with_suffix(''))+ '-anatomical-structure.json', 'r') as f:
        data = json.load(f)
    h,w = mask.shape
    cortex_rec = [r for r in data if r['properties']['classification']['name'] == 'Cortex']
    cortex_poly = np.array(cortex_rec[0]['geometry']['coordinates']).astype(np.int32)
    buf = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(buf, cortex_poly, 1)
    mask = mask * buf
    return mask

def dump_to_csv(results, dst_path, threshold):
    df = pd.read_csv('input/hm/sample_submission.csv', index_col='id')
    for k, v in results.items():
        df.loc[k.stem] = v
    df.to_csv(dst_path + f'submission__{int(threshold)}.csv')


def mask_q_writer(q, H, W, batch_size, total_blocks, root, pad, result):
    import infer

    do_inference = infer.get_infer_func(root, use_tta=True)
    mask = np.zeros((1,H,W)).astype(np.uint8)
    count = 0
    batch = []
    
    while count < total_blocks:
        if len(batch) == batch_size:
            blocks = [b[0] for b in batch]
            block_cds = [b[1] for b in batch]

            block_masks = infer_blocks(blocks, do_inference)
            [paste_crop(mask, block_mask, block_cd, pad) for block_mask, block_cd, block in zip(block_masks, block_cds, blocks)]
            batch = []
        else:
            batch.append(q.get())
            count+=1
            q.task_done()

    if batch:
        blocks = [b[0] for b in batch]
        block_cds = [b[1] for b in batch]
        block_masks = infer_blocks(blocks, do_inference)
        [paste_crop(mask, block_mask, block_cd, pad) for block_mask, block_cd, block in zip(block_masks, block_cds, blocks) if block.mean() > 0]
        #print('Drop last batch', len(batch))
        batch = []
    
    result['mask'] = mask
    return


def launch_mpq(img_name, model_folder, batch_size, block_size, pad, num_processes, qsize):
    m = mp.Manager()
    result = m.dict()
    q = m.Queue(maxsize=qsize)
    _, (H,W), _ = get_basics_rasterio(img_name)
    cds = list(generate_block_coords(H, W, block_size=(block_size,block_size)))
    total_blocks = len(cds)
        
    reader_args = [(q, img_name, part_cds, pad) for part_cds in chunkify(cds, num_processes)]
    reader = partial(mp_func_wrapper, image_q_reader)
    
    writer = partial(mp_func_wrapper, mask_q_writer)
    writer_p = mp.Process(target=writer, args=((q,H,W, batch_size, total_blocks, model_folder, pad, result),))
    writer_p.start()        
    
    with mp.Pool(num_processes) as p:    
        g = tqdm(p.imap_unordered(reader, reader_args), total=len(reader_args))
        for _ in g:
            pass
    
    writer_p.join()
    writer_p.terminate()
    mask = result['mask']
    print(mask.shape, mask.dtype, mask.max())
    return mask[0]

def gpu_select(gpus):
    if isinstance(gpus, list):
        return gpus[0] # local run

    # MP run from run_inference
    time.sleep(random.random())
    gpu = None
    while True:
        time.sleep(1)
        for k,v in gpus.items():
            if not v:
                gpu = k
                gpus[k] = True
                break
        if gpu is not None: break
    return gpu
            

def start(img_name, gpus, model_folder, threshold, results):
    random.seed(hash(str(img_name)))
    gpu = gpu_select(gpus)
    logger.log('DEBUG', f'Starting inference {img_name} on GPU {gpu}')
    logger.log('DEBUG', f'{gpu}, {gpus}')

    os.environ['CPL_LOG'] = '/dev/null'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    dst = Path('output/predicts')
    block_size = 2048
    batch_size = 8
    pad = 1024
    num_processes = 4
    qsize = 24
    save_predicts = True
    join_predicts = False

    mask = launch_mpq(str(img_name),
                        model_folder, 
                        batch_size=batch_size, 
                        block_size=block_size, 
                        pad=pad, 
                        num_processes=num_processes, 
                        qsize=qsize)

    #mask = filter_mask(mask, img_name)
    mask[mask<threshold] = 0
    gpus[gpu] = False

    if save_predicts:
        out_name = dst/'masks'/img_name.name
        utils.save_tiff_uint8_single_band(mask, str(out_name))
        if join_predicts:
            merge_name = dst/'merged'/img_name.name
            utils.tiff_merge_mask(img_name, str(out_name), merge_name)

    print('Done , to RLE:')
    mask = mask.clip(0,1)
    rle = rle2tiff.mask2rle(mask)
    results[img_name] = rle


if __name__ == '__main__':
    src = Path('input/hm/test')
    img_names = src.glob('*.tiff')
    img_names = list(img_names)
    img_names = [img_names[i] for i in [2,0,4,1,3]]#aa first
    print(list(img_names))

    model_folder = 'output/ma34folds/'
    #model_folder = 'output/2021_Mar_21_18_52_11_PAMBUH/'
    threshold = int(.4 * 255)#105
    gpus = ['0']
    results = {}

    for img_name in img_names:
        print(f'Creating mask for {img_name}')
        start(model_folder, gpus, img_name, threshold, results)

    dump_to_csv(results, model_folder, threshold)

