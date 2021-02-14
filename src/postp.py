import os
import sys
from pathlib import Path
import multiprocessing as mp
from functools import partial

import numpy as np
import rasterio as rio
from tqdm.auto import tqdm

import utils
import rle2tiff

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
    
    nYValid, nXValid = h, w
    for X in range(nXBlocks):
        #if X == nXBlocks - 1: nXValid = W - X * h
        cx = X * h
        nYValid = w
        for Y in range(nYBlocks):
            #if Y == nYBlocks - 1: nYValid = H - Y *w
            cy = Y * w
            yield cy, cx, nYValid, nXValid

            
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
def paste_center_crop(src, part, block):
    C,H,W = src.shape
    y,x,h,w = block
    
    h, w = min(h, H-y), min(w, W-x)  
    y, x = get_shift_center(part, h,w)
    part = crop(part, y, x, h, w)
    paste(src, part, *block)

def filter_block_out(block): return block.max()<5
def infer_blocks(blocks, do_inference):
    blocks = do_inference(blocks).cpu().numpy()    
    blocks = (255*blocks).astype(np.uint8)
    return blocks
 

def image_q_reader(q, p, cds, pad):
    for block_cd in cds:
        padded_block_cd = pad_block(*block_cd, pad)
        block = crop_rio(p, *(padded_block_cd))
        #print(f'Read block {block.shape, block.dtype}')
        #if filter_block_out(block): continue
        q.put((block, block_cd))
        
def mask_q_writer(q, H, W, total_blocks, root, result):
    do_inference = infer.get_infer_func(root)
    mask = np.zeros((1,H,W)).astype(np.uint8)
    count = 0
    bs = 8
    batch = []
    
    while count < total_blocks:
        try:
            if len(batch) == bs:
                #print('Drop batch', len(batch))
                blocks = [b[0] for b in batch]
                block_cds = [b[1] for b in batch]

                block_masks = infer_blocks(blocks, do_inference)#.squeeze(0)
                #block_masks = [np.expand_dims(b[0], 0) for b in blocks]
                [paste_center_crop(mask, block_mask, block_cd) for block_mask, block_cd in zip(block_masks, block_cds)]
                batch = []
            else:
                item = q.get()
                batch.append(item)
                #print('Add batch', len(batch), count, total_blocks, q.qsize())
                count+=1
                q.task_done()
            
        except mp.TimeoutError:
            print("timeout, quit.")
            break
        except Exception as e:
            print(e)
            break
    if batch:
        print('Drop last batch', len(batch))
        batch = []
    
    result['mask'] = mask
    return

def mp_func_wrapper(func, args): return func(*args)
def chunkify(l, n): return [l[i:i + n] for i in range(0, len(l), n)]


def launch_mpq(p, model_folder, block_size=512, pad=16, num_processes=1, show_tqdm=True, qsize=12):
    m = mp.Manager()
    result = m.dict()
    q = m.Queue(maxsize=qsize)
    _, (H,W), _ = get_basics_rasterio(p)
    cds = list(generate_block_coords(H, W, block_size=(block_size,block_size)))
        
    n_chunks = num_processes#1 + len(cds) // num_processes
    reader_args = [(q, p, part_cds, pad) for part_cds in chunkify(cds, n_chunks)]
    #assert len(reader_args) == num_processes
    reader = partial(mp_func_wrapper, image_q_reader)
    
    total_blocks = len(reader_args)
    writer = partial(mp_func_wrapper, mask_q_writer)

    writer_p = mp.Process(target=writer, args=((q,H,W,total_blocks, model_folder, result),))
    writer_p.start()        
    
    
    if show_tqdm: pbar = tqdm(total=len(reader_args))
    with mp.Pool(num_processes) as p:    
        g = p.imap_unordered(reader, reader_args)
        for _ in g:
            if show_tqdm: pbar.update()    
    
    print('joining')
    writer_p.join()
    mask = result['mask']
    print(mask.shape, mask.dtype, mask.max())
    return mask


if __name__ == '__main__':
    import infer
    os.environ['CPL_LOG'] = '/dev/null'

    model_folder = 'output/2021_Feb_12_20_08_44_PAMBUH/'
    block_size = 2048
    pad = 0
    threshold = 200
    save_predicts = True
    join_predicts = True

    src = Path('input/hm/train')
    img_names = src.glob('*.tiff')
    dst = Path('output/predicts')

    for img_name in img_names:
        print(f'Creating mask for {img_name}')
        mask = launch_mpq(str(img_name), model_folder, block_size=block_size, pad=pad, num_processes=1)
        mask[mask<threshold] = 0
        mask = mask.clip(1)

        rle2tiff.mask2rle(mask)
        
        if save_predicts:
            out_name = dst/img_name.name
            utils.save_tiff_uint8_single_band(mask[0], str(out_name))
            if join_predicts:
                merge_name = dst/'merged'/img_name.name
                utils.tiff_merge_mask(img_name, str(out_name), merge_name)

        break
    


