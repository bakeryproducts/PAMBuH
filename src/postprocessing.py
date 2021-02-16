import csv
from pathlib import Path
from typing import NoReturn, Union
from datetime import datetime
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rasterio

import utils
import rle2tiff

def get_chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def read_and_process_img(path: str, do_infer, block_size: int = 2048, crop_size: int = 2000, batch_size: int = 4, threshold : float = 0.5) -> torch.tensor:
    """ Make prediction mask of big tiff by little crop

    Args:
        path (str): [Path to image]
        do_infer (function: [list of tensor [CxHxW] -> tensor [Bx1xHxW]]): infer function
        block_size (int): [size of block for infer function]. Defaults to 2048.
        crop_size (int): [size of crop, that will be connected in filal mask]. Defaults to 2000.
        batch_size (int): [size of batch for infer function]. Defaults to 4.
        threshold (float, 0..1). Defaults to 0.5.
    Returns:
        np.array (uint8): [infer mask]: [HxW]
    """
    assert block_size >= crop_size, (block_size, crop_size)
    fd, (h, w), channel = utils.get_basics_rasterio(path)
    pad = (block_size - crop_size)//2
    rows = []
    for y in tqdm(range(-pad, h, crop_size), desc='rows'):
        row, zeros_idx = [], []
        for i, x in enumerate(tqdm(range(-pad, w, crop_size), desc='columns')):
            pad_x = (pad if x < 0 else 0, x + block_size - w if x + block_size > w else 0)
            pad_y = (pad if y < 0 else 0, y + block_size - h if y + block_size > h else 0)
            pad_chw = ((0, 0), pad_y, pad_x)
            block = utils.get_tiff_block(fd, x, y, block_size)
            block = np.pad(block, pad_chw, 'constant', constant_values=0)
            if block.max() > 0:
                row.append(block)
            else:
                zeros_idx.append(i)
        if row:
            masks = []
            for chunk in get_chunk(row, batch_size):
                mask = do_infer(chunk).to('cpu')[:, 0, pad:-pad, pad:-pad]
                mask = np.array(mask*255, dtype=np.uint8)
                masks.append(mask)
            masks = np.concatenate(masks, axis=0)
            for i in zeros_idx:
                masks = np.insert(masks, i, np.zeros((crop_size, crop_size), dtype=np.uint8), axis=0)
            masks = np.concatenate(list(masks), axis=1)
        else:
            masks = np.zeros((crop_size, w), dtype=np.uint8)
        rows.append(masks[:, :w])
    mask = np.concatenate(rows, axis=0)[:h, :w]
    assert mask.shape == (h, w), (mask.shape, (h, w))
    return np.array(mask > 256*threshold, dtype=np.uint8)

def save_join_predict(img_name, mask, path_dst):
    """
        img_name (str): path to image
        mask (np.array): uint8 infer mask [0, 1]
        path_dst(str): path to save
    """
    assert mask.max() <= 1 + 1e-6
    fd, shape, channel = utils.get_basics_rasterio(img_name)
    h, w = shape
    img = utils.get_tiff_block(fd, 0, 0, w, h, 3)
    img[1] = mask*200
    dst = rasterio.open(path_dst, 'w', driver='GTiff', height=h, width=w, count=3, dtype=np.uint8)
    dst.write(img, [1,2,3]) # 3 bands
    dst.close()
    del dst

def _plot_img(img: np.ndarray) -> NoReturn:
    print(img.shape)
    if len(img.shape) == 2:
        plt.imshow(np.array(img))
    else:
        plt.imshow(np.array(img).transpose(1, 2, 0))
    plt.show()

def postprocess_test_folder(infer_func, src_folder : str, dst_folder : str, threshold : float=0.5, save_predicts : bool = True, join_predicts : bool = True) -> NoReturn:
    """
    Args:
        infer_func(list(img1, img2, ...) -> [BxCxHxW]
        src_folder - folder with test images
        dst_folder - folder to save output (RLE and predictions)
        threshold (float, 0..1). Defaults to 0.5.
    """

    imgs_name = Path(src_folder).glob('*.tiff')
    dst_folder = Path(dst_folder) / str(datetime.now()).replace(' ', '_')
    dst_folder.mkdir(parents=True, exist_ok=True)
    with open(dst_folder / 'sample_submission.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id', 'predicted'])
        for img_name in tqdm(imgs_name, desc='Test images', leave=False):
            mask = read_and_process_img(img_name, infer_func, threshold=threshold)
            if save_predicts:
                utils.save_tiff_uint8_single_band(mask, dst_folder / img_name.name)
            if join_predicts:
                save_join_predict(img_name, mask, dst_folder / ('join_' + img_name.name))
            spamwriter.writerow([img_name.stem, rle2tiff.mask2rle(mask)])
    print(f'save to {dst_folder / "sample_submission.csv"}')
