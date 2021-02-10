import numpy as np
import torch
import torch.nn.functional as F

from typing import NoReturn, Union

def read_frame(fd, x, y, h, w=None, c=3):
	if w is None: w = h
	img = fd.read(list(range(1, c+1)), window=Window(x, y, h, w))
	return torch.ByteTensor(img)

def block_reader(path, inf, size=512, infer_list_flg=False):
	fd, shape, channel = get_basics_rasterio(path)
	
	rows = []
	for ny in tqdm(range(-size//4, shape[0], size//2), desc='rows'):
		pad=(size//4, size - shape[1]%size)
		if ny < 0:
			pad = (size//4, size - shape[1]%size, size//4, 0)
		elif shape[0]-ny < size:
			pad = (size//4, size - shape[1]%size, 0, size + ny - shape[0])
		row = read_frame(fd, 0, ny, shape[1], size)
		row_img = F.pad(row, pad=pad, mode='constant', value=0)
		row_img = row_img.unsqueeze(0)
		left_i = torch.split(row_img, size, dim=3)[:-1]
		right_i = torch.split(row_img[:, :, :, size//4:], size, dim=3)

		imgs_batch = []
		for i in range(len(left_i)):
			imgs_batch.extend([left_i[i], right_i[i]])
		if infer_list_flg:
			infer_batch = inf(imgs_batch)
		else:
			infer_batch = inf(torch.cat(imgs_batch, 0))
		infer_batch = infer_batch[:, :, size//4:-size//4, size//4:-size//4]
		rows.append(torch.cat([i for i in infer_batch], 2).squeeze(0))
	return torch.cat(rows, 0)[:shape[0], :shape[1]]

def prt(img):
	print(img.shape)
	if len(img.shape) == 2:
		plt.imshow(np.array(img))
	else:
		plt.imshow(np.array(img).transpose(1, 2, 0))
	plt.show()

def save_tiff_uint8_single_band(img, path):
	if isinstance(img, torch.Tensor):
		img = np.array(img)
	elif not isinstance(img, numpy.ndarray):
		raise TypeError(f'Want torch.Tensor or numpy.ndarray, but got {type(img)}')
	assert img.dtype == np.uint8
	h, w = img.shape
	dst = rasterio.open(path, 'w', driver='GTiff', height=h, width=w, count=1, nbits=1, dtype=np.uint8)
	dst.write(img, 1) # 1 band
	dst.close()
	print(f'Save to {path}')
	del dst

def postprocess(infer_func, src_folder, dst_folder, save_predicts=True):
	"""
	infer_func([BxCxHxV]) -> [BxCxHxV]
	src_folder - folder with test images
	dst_folder - folder to save output (RLE and predictions)
	"""
	imgs_name = Path(src_folder).glob('*.tiff')
	create_dir(dst_folder)
	for img_name in tqdm(imgs_name, desc='Images', leave=False):
		img = block_reader(img_name, infer_func)
		save_tiff_uint8_single_band(img, Path(dst_folder) / img_name.name)

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
