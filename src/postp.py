import os
import json
import sys
from pathlib import Path
import multiprocessing as mp
from functools import partial

import cv2
import pandas as pd
import numpy as np
import rasterio as rio
from tqdm import tqdm

import utils
import rle2tiff

import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def _count_blocks(dims, block_size=(256, 256)):
    nXBlocks = (int)((dims[0] + block_size[0] - 1) / block_size[0])
    nYBlocks = (int)((dims[1] + block_size[1] - 1) / block_size[1])
    return nXBlocks, nYBlocks

def get_basics_rasterio(name):
    file = rio.open(str(name))
    return file, file.shape, file.count

def generate_block_coords(H, W, block_size=(128,128)):
    h,w = block_size
    nYBlocks = (int)((H + h - 1) / h)
    nXBlocks = (int)((W + w - 1) / w)
    
    for X in range(nXBlocks):
        cx = X * h
        for Y in range(nYBlocks):
            cy = Y * w
            yield cy, cx, h, w

            
def get_shift_center(block, h,w): return block.shape[1]//2 -h//2, block.shape[2]//2 -w//2
def pad_block(y,x,h,w, pad): return np.array([y-pad, x-pad, h+pad, w+pad])
def crop(src, y,x,h,w): return src[..., y:y+h, x:x+w]
def crop_rio(p, y,x,h,w, bands=(1,2,3)):
    ds = rio.open(p)
    block = ds.read(bands, window=((y,y+h),(x,x+w)), boundless=True)
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
    blocks = (255*blocks).astype(np.uint8)
    return blocks
 
def image_q_reader(q, p, cds, pad):
    for block_cd in cds:
        padded_block_cd = pad_block(*block_cd, pad)
        block = crop_rio(p, *(padded_block_cd))
        q.put((block, block_cd))
        
def mask_q_writer(q, H, W, total_blocks, root, pad, result):
    do_inference = infer.get_infer_func(root, use_tta=True)
    mask = np.zeros((1,H,W)).astype(np.uint8)
    count = 0
    BATCH_SIZE = 8
    batch = []
    
    while count < total_blocks:
        try:
            if len(batch) == BATCH_SIZE:
                #print('Drop batch', len(batch))
                blocks = [b[0] for b in batch]
                block_cds = [b[1] for b in batch]

                block_masks = infer_blocks(blocks, do_inference)
                [paste_crop(mask, block_mask, block_cd, pad) for block_mask, block_cd, block in zip(block_masks, block_cds, blocks) if block.mean() > 0]
                batch = []
            else:
                batch.append(q.get())
                count+=1
                q.task_done()
        except mp.TimeoutError:
            print("timeout, quit.")
            break
        except Exception as e:
            print(e)
            break

    if batch:
        blocks = [b[0] for b in batch]
        block_cds = [b[1] for b in batch]
        block_masks = infer_blocks(blocks, do_inference)
        [paste_crop(mask, block_mask, block_cd, pad) for block_mask, block_cd, block in zip(block_masks, block_cds, blocks) if block.mean() > 0]
        #print('Drop last batch', len(batch))
        batch = []
    
    result['mask'] = mask
    return

def mp_func_wrapper(func, args): return func(*args)
def chunkify(l, n): return [l[i:i + n] for i in range(0, len(l), n)]

def launch_mpq(img_name, model_folder, block_size=512, pad=16, num_processes=1, qsize=24):
    m = mp.Manager()
    result = m.dict()
    q = m.Queue(maxsize=qsize)
    _, (H,W), _ = get_basics_rasterio(img_name)
    cds = list(generate_block_coords(H, W, block_size=(block_size,block_size)))
    total_blocks = len(cds)
        
    reader_args = [(q, img_name, part_cds, pad) for part_cds in chunkify(cds, num_processes)]
    reader = partial(mp_func_wrapper, image_q_reader)
    
    writer = partial(mp_func_wrapper, mask_q_writer)
    writer_p = mp.Process(target=writer, args=((q,H,W,total_blocks, model_folder, pad, result),))
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

def filter_mask(mask, name):
    with open(str(name.with_suffix(''))+ '-anatomical-structure.json', 'r') as f:
        data = json.load(f)
    h,w = mask.shape
    cortex_rec = [r for r in data if r['properties']['classification']['name'] == 'Cortex']
    cortex_poly = np.array(cortex_rec[0]['geometry']['coordinates'])
    buf = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(buf, np.array(cortex_poly), 1)
    mask = mask * buf
    return mask

def global_shift_mask(name, maskpred1):
    """
    applies a global shift to a mask by padding one side and cropping from the other
    """
    TARGET_ID = 'afa5e8098'
    y_shift = -40
    x_shift = -24
    if name != TARGET_ID: return maskpred1

    if y_shift <0 and x_shift >=0:
        maskpred2 = np.pad(maskpred1, [(0,abs(y_shift)), (abs(x_shift), 0)], mode='constant', constant_values=0)
        maskpred3 = maskpred2[abs(y_shift):, :maskpred1.shape[1]]
    elif y_shift >=0 and x_shift <0:
        maskpred2 = np.pad(maskpred1, [(abs(y_shift),0), (0, abs(x_shift))], mode='constant', constant_values=0)
        maskpred3 = maskpred2[:maskpred1.shape[0], abs(x_shift):]
    elif y_shift >=0 and x_shift >=0:
        maskpred2 = np.pad(maskpred1, [(abs(y_shift),0), (abs(x_shift), 0)], mode='constant', constant_values=0)
        maskpred3 = maskpred2[:maskpred1.shape[0], :maskpred1.shape[1]]
    elif y_shift < 0 and x_shift < 0:
        maskpred2 = np.pad(maskpred1, [(0, abs(y_shift)), (0, abs(x_shift))], mode='constant', constant_values=0)
        maskpred3 = maskpred2[abs(y_shift):, abs(x_shift):]
    return maskpred3

if __name__ == '__main__':
    os.environ['CPL_LOG'] = '/dev/null'
    import infer

    #model_folder = 'output/2021_Feb_18_23_33_58_PAMBUH/'
    #model_folder = 'output/2021_Feb_20_00_13_39_PAMBUH/'
    model_folder = 'output/2021_Feb_23_17_04_53_PAMBUH/'

    block_size = 2048
    pad = 512
    threshold = 160
    num_processes = 8
    qsize = 16
    save_predicts = False
    join_predicts = True


    src = Path('input/hm/test')
    img_names = src.glob('*.tiff')
    dst = Path('output/predicts')
    df = pd.read_csv('input/hm/sample_submission.csv', index_col='id')
    img_names = list(img_names)
    print(list(img_names))

    for img_name in img_names:
        #img_name = img_names[-2]
        print(f'Creating mask for {img_name}')
        mask = launch_mpq(str(img_name), model_folder, block_size=block_size, pad=pad, num_processes=num_processes, qsize=qsize)
        #mask = filter_mask(mask, img_name)
        mask[mask<threshold] = 0
        mask = global_shift_mask(img_name.stem, mask)

        if save_predicts:
            out_name = dst/'masks'/img_name.name
            utils.save_tiff_uint8_single_band(mask, str(out_name))
            if join_predicts:
                merge_name = dst/'merged'/img_name.name
                utils.tiff_merge_mask(img_name, str(out_name), merge_name)

        print('Done , to RLE:')
        mask = mask.clip(0,1)
        rle = rle2tiff.mask2rle(mask)
        df.loc[img_name.stem] = rle
        #break

    df.to_csv(model_folder + 'submission.csv')
    


