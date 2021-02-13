#from tqdm import tqdm
import csv
from tqdm.notebook import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import NoReturn, Union

import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import get_basics_rasterio, get_tiff_block, save_tiff_uint8_single_band
from rle2tiff import mask2rle

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_and_process_img(path: str, do_infer, block_size: int = 2048, crop_size: int = 2000, batch_size: int = 4, device : str = 'cuda') -> np.ndarray:
	""" Giga Function doing everything. made by galayda 

	Args:
		path (str): [description]
		do_infer ([type]): [description]
		block_size (int, optional): [description]. Defaults to 2048.
		crop_size (int, optional): [description]. Defaults to 2000.
		batch_size (int, optional): [description]. Defaults to 4.

	Returns:
		np.ndarray: [description]
	"""    
	assert block_size >= crop_size, (block_size, crop_size)
	fd, (h, w), channel = get_basics_rasterio(path)
	print(h, w)
	pad = (block_size - crop_size)//2
	rows = []
	for y in tqdm(range(-pad, h, crop_size), desc='rows'):
		row, zeros_idx = [], []
		for i, x in enumerate(tqdm(range(-pad, w, crop_size), desc='columns')):
			pad_x = (pad if x < 0 else 0, x + block_size - w if x + block_size > w else 0)
			pad_y = (pad if y < 0 else 0, y + block_size - h if y + block_size > h else 0)
			pad_chw = ((0, 0), pad_y, pad_x)
			block = get_tiff_block(fd, x, y, block_size)
			pad_block = np.pad(block, pad_chw, 'constant', constant_values=0)
			if pad_block.max() > 0:
				row.append(pad_block)
			else:
				zeros_idx.append(i)
		if row:
			nozero_masks = []
			for chunk in chunks(row, batch_size):
				nozero_masks.append(do_infer(chunk)[:, :, pad:-pad, pad:-pad])
			nozero_masks = torch.cat(nozero_masks, 0)
			device = nozero_masks.device
			masks_list = [i.squeeze(0) for i in nozero_masks]
			for i in zeros_idx:
				masks_list.insert(i, torch.zeros((crop_size, crop_size)).to(device))
			mask_row = torch.cat(masks_list, 1)[:, :w]
		else:
			mask_row = torch.zeros((crop_size, w)).to(device)
		rows.append(mask_row)
	mask = torch.cat(rows, 0)[:h, :w]
	assert mask.shape == (h, w)
	return mask

def float_mask2_01uint8(mask, threshold=0.5):
	return torch.as_tensor(mask > threshold, dtype=torch.int8)

def _plot_img(img: np.ndarray) -> NoReturn:
	print(img.shape)
	if len(img.shape) == 2:
		plt.imshow(np.array(img))
	else:
		plt.imshow(np.array(img).transpose(1, 2, 0))
	plt.show()

def postprocess_test_folder(infer_func, src_folder : str, dst_folder : str) -> NoReturn:
	"""
	infer_func(list(img1, img2, ...) -> [BxCxHxW]
	src_folder - folder with test images
	dst_folder - folder to save output (RLE and predictions)
	"""
	imgs_name = Path(src_folder).glob('*.tiff')
	dst_folder = Path(dst_folder)
	dst_folder.mkdir(parents=True, exist_ok=True)
	with open(dst_folder / 'sample_submission.csv', 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['id', 'predicted'])
		for img_name in tqdm(imgs_name, desc='Test images', leave=False):
			mask = read_and_process_img(img_name, infer_func)
			mask01 = np.uint8(float_mask2_01uint8(mask).to('cpu'))
			#save_tiff_uint8_single_band(mask01, dst_folder / img_name.name)
			spamwriter.writerow([img_name.stem, mask2rle(mask01.T)])
	print(f'save to {dst_folder / "sample_submission.csv"}')
