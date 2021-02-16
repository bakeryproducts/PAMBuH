import csv
#from tqdm import tqdm
from tqdm.notebook import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import NoReturn, Union
from datetime import datetime
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


def read_and_process_img(path: str, do_infer, block_size: int = 2048, crop_size: int = 2000, batch_size: int = 4, device : str = 'cuda') -> torch.tensor:
	""" Make prediction mask of big tiff by little crop

	Args:
		path (str): [Path to image]
		do_infer (function: [list of tensor [CxHxW] -> tensor [Bx1xHxW]]): infer function
		block_size (int): [size of block for infer function]. Defaults to 2048.
		crop_size (int): [size of crop, that will be connected in filal mask]. Defaults to 2000.
		batch_size (int): [size of batch for infer function]. Defaults to 4.

	Returns:
		torch.tensor: [infer mask]: [HxW]
	"""    
	assert block_size >= crop_size, (block_size, crop_size)
	fd, (h, w), channel = get_basics_rasterio(path)
	print(f'input shape (h, w): {h, w}')
	pad = (block_size - crop_size)//2
	rows = []
	for y in tqdm(range(-pad, h, crop_size), desc='rows'):
		row, zeros_idx = [], []
		for i, x in enumerate(tqdm(range(-pad, w, crop_size), desc='columns')):
			pad_x = (pad if x < 0 else 0, x + block_size - w if x + block_size > w else 0)
			pad_y = (pad if y < 0 else 0, y + block_size - h if y + block_size > h else 0)
			pad_chw = ((0, 0), pad_y, pad_x)
			block = get_tiff_block(fd, x, y, block_size)
			block = np.pad(block, pad_chw, 'constant', constant_values=0)
			if block.max() > 0:
				row.append(block)
			else:
				zeros_idx.append(i)
		if row:
			masks = []
			for chunk in chunks(row, batch_size):
				masks.append(do_infer(chunk)[:, :, pad:-pad, pad:-pad])
			masks = torch.cat(masks, 0)
			device = masks.device
			masks_list = [i.squeeze(0) for i in masks]
			for i in zeros_idx:
				masks_list.insert(i, torch.zeros((crop_size, crop_size)).to(device))
			masks = torch.cat(masks_list, 1)[:, :w]
		else:
			masks = torch.zeros((crop_size, w)).to(device)
		rows.append(masks)
	mask = torch.cat(rows, 0)[:h, :w]
	assert mask.shape == (h, w)
	return mask

def float_mask2_01uint8(mask : torch.tensor, threshold : float=0.5):
	""" Convert float mask (0..1) to binary mask [0, 1]

	Args:
		mask (torch.tensor): [Mask, any element of that beetween 0 and 1].
		threshold (float, 0..1). Defaults to 0.5.

	Returns:
		torch.tensor: infer mask (int8)
	"""    
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
	dst_folder = Path(dst_folder) / str(datetime.now()).replace(' ', '_')
	dst_folder.mkdir(parents=True, exist_ok=True)
	with open(dst_folder / 'sample_submission.csv', 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['id', 'predicted'])
		for img_name in tqdm(imgs_name, desc='Test images', leave=False):
			mask = read_and_process_img(img_name, infer_func)
			mask01 = np.uint8(float_mask2_01uint8(mask).to('cpu'))
			save_tiff_uint8_single_band(mask01, dst_folder / img_name.name)
			spamwriter.writerow([img_name.stem, mask2rle(mask01)])
	print(f'save to {dst_folder / "sample_submission.csv"}')
