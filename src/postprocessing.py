#from tqdm import tqdm
from tqdm.notebook import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import NoReturn, Union
import matplotlib.pyplot as plt
from utils import get_tiff_block, save_tiff_uint8_single_band, jdump
from sampler import get_basics_rasterio
from rle2tiff import mask2rle

def read_and_process_img(path : str, do_infer, block_size : int = 512, crop_size : int = 500, batch_size : int = 4) -> np.ndarray:
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
			masks_list = [i.squeeze(0) for i in nozero_masks]
			for i in zeros_idx:
				masks_list.insert(i, torch.zeros((crop_size, crop_size)))
			mask_row = torch.cat(masks, 1)
		else:
			mask_row = torch.zeros((crop_size, w))
		rows.append(mask_row)
	mask = np.uint8((torch.cat(rows, 0)[:h, :w]*255).to('cpu'))
	assert mask.shape == (h, w)
	return mask

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
	for img_name in tqdm(imgs_name, desc='Test images', leave=False):
		mask = read_and_process_img(img_name, infer_func)
		save_tiff_uint8_single_band(mask, dst_folder / img_name.name)
		jdump(mask2rle(mask), dst_folder / img_name.stem + '.json')
			
class SmoothTiles:
	def __init__(self, window_fun: str ='triangle', cuda: bool = True) -> NoReturn:
		self.window_fun_name = window_fun
		self.get_window_fun()
		self.cuda = cuda and torch.cuda.is_available()

	def __call__(self, img_main: torch.float, img_sub: torch.float) -> torch.float:
		return self.merge(img_main, img_sub)

	def __repr__(self) -> NoReturn:
		return f'Name of window funcion: {self.window_fun_name}'

	def get_window_fun(self) -> NoReturn:
		self.window_fun = {
			'triangle': self.triangle,
			'gauss': self.gauss,
			'spline': self.spline
		}[self.window_fun_name]

	def spline(self, window_size: int, power: float = 2) -> np.array:
		intersection = int(window_size/4)
		wind_outer = (abs(2*(self.triangle(window_size))) ** power)/2
		wind_outer[intersection:-intersection] = 0
		
		wind_inner = 1 - (abs(2*(self.triangle(window_size) - 1)) ** power)/2
		wind_inner[:intersection] = 0
		wind_inner[-intersection:] = 0
		
		wind = wind_inner + wind_outer
		return wind

	def gauss(self, window_size: int, std : float = None) -> np.array:
		if std is None: std = window_size/4
		n = np.arange(0, window_size) - (window_size - 1.0) / 2.0
		return np.exp(-n**2/(2*std*std))

	def triangle(self, window_size: int) -> np.array:
		n = np.arange(1, (window_size + 1) // 2 + 1)
		if window_size % 2 == 0:
			w = (2 * n - 1.0) / window_size
			w = np.r_[w, w[::-1]]
		else:
			w = 2*n/(window_size + 1.0)
			w = np.r_[w, w[-2::-1]]
		return w

	def window_2D(self, window_size: int, window=None) -> np.array:
		if window is None: window=self.window_fun
		wind = window(window_size)
		wind = np.expand_dims(wind, 1)
		wind = wind * wind.transpose(1, 0)
		return wind

	def merge(self, img_main: np.array, img_sub: np.array) -> np.array:
		n = int(np.sqrt(img_sub.shape[-1]))
		m = img_sub.shape[0]
		img_sub = torch.reshape(img_sub, (img_sub.shape[0], img_sub.shape[1], n, n))
		top = torch.cat((img_sub[:,:,:-1,:-1], img_sub[:,:,1:,:-1]), axis=0)
		down = torch.cat((img_sub[:,:,:-1,1:], img_sub[:,:,1:,1:]), axis=0)
		mean = torch.cat((top, down), axis=1)
		img_sub = torch.reshape(mean[m//2:-m//2, m//2:-m//2], (img_sub.shape[0], img_sub.shape[1], -1))
		main = self.window_2D(img_main.shape[0])
		tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
		main = tensor(np.repeat(np.expand_dims(main, 2), img_sub.shape[-1], axis=2))
		return main*img_main + (1-main)*img_sub
