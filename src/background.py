import os
import sys
import json

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets as widgets

import utils

def read_with_scale(img_path, scale=10):
    fd, (h, w), channel = utils.get_basics_rasterio(img_path)
    img = utils.get_tiff_block(fd, 0, 0, w, h)
    img = np.moveaxis(img, 0, -1)
    img = cv2.resize(img, dsize=(w//scale, h//scale), interpolation=cv2.INTER_NEAREST)
    return img

def rescale_images(src_folder, dst_folder, scale=10):
    scale_names = []
    src_folder, dst_folder = Path(src_folder), Path(dst_folder)
    for img_name in src_folder.glob('*.tiff'):
        img_name = next(Path(src_folder).glob('*.tiff'))
        img = read_with_scale(img_name, scale)
        scale_name = dst_folder / (img_name.stem + f'_scale_{scale}.png')
        cv2.imwrite(str(scale_name), img)
        scale_names.append(scale_name)
    return scale_names

class Markup:
    def __init__(self, img_name, dst_path, scale=10):
        self.scale = scale
        self.img_name = img_name
        self.dst_path = Path(dst_path)

    def _onclick(self, event):
        self.coord.append([int(event.xdata), int(event.ydata)])
        if len(self.coord) > 1:
            xs, ys = zip(*self.coord)
            plt.plot(xs, ys, 'g')
            
    def _on_button_clicked1(self, b):
        self.coord = self.coord + [self.coord[0]]
        xs, ys = zip(*self.coord)
        plt.plot(xs, ys, 'g')
        self.all_coords.append(self.coord)
        self.coord  = []

    def _on_button_clicked2(self, b):
        self._dump_background_coords()

    def _dump_background_coords(self):
        data = []
        for coords in self.all_coords:
            data.append({"geometry": {"type": "Polygon", "coordinates": [[[x*self.scale, y*self.scale] for x, y in coords]]}})
        backgrounds_name = self.img_name.stem + '_background.json'
        with open(self.dst_path / backgrounds_name, 'w') as outfile:
            json.dump(data, outfile)

    def markup_background(self):
        self.coord = []
        self.all_coords = []
        fig = plt.figure()
        img = mpimg.imread(str(self.img_name))

        cid = fig.canvas.mpl_connect('button_press_event', self._onclick)
        imgplot = plt.imshow(img)
        plt.show()

        button1 = widgets.Button(description='Write polygon')
        button2 = widgets.Button(description='Dump_backgrounds')
        button1.on_click(self._on_button_clicked1)
        button2.on_click(self._on_button_clicked2)
        return widgets.VBox([button1, button2])
