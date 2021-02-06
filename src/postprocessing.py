import torch
import numpy as np

from typing import NoReturn, Union

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
