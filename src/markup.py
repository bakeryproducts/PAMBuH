import os
import sys
import json

from tqdm.auto import tqdm
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
    for img_name in tqdm(src_folder.glob('*.tiff')):
        img = read_with_scale(img_name, scale)
        scale_name = dst_folder / (img_name.stem + f'_scale_{scale}.png')
        cv2.imwrite(str(scale_name), img)
        scale_names.append(scale_name)
    return scale_names

class Markup:
    def __init__(self, src_name, dst_name, scale=10):
        self.scale = scale
        self.src_name = Path(src_name)
        self.dst_name = Path(dst_name)

    def _onclick(self, event):
        self.polygon.append([int(event.xdata), int(event.ydata)])
        if len(self.polygon) > 1:
            self._plot_polygon(self.polygon)

    def _on_draw_button_clicked(self, b):
        self.polygon = self.polygon + [self.polygon[0]]
        self._plot_polygon(self.polygon)
        self.all_polygons.append(self.polygon)
        self.polygon  = []

    def _plot_polygon(self, polygon):
        xs, ys = zip(*polygon)
        plt.plot(xs, ys, color='green')

    def _on_dump_button_clicked(self, b):
        self._dump_polygons()

    def _dump_polygons(self):
        data = []
        for polygon in self.all_polygons:
            data.append({"geometry": {"type": "Polygon", "coordinates": [[[x*self.scale, y*self.scale] for x, y in polygon]]}})
        with open(self.dst_name, 'w') as f:
            json.dump(data, f)

    def markup_polygons(self):
        self.polygon = []
        self.all_polygons = []
        fig = plt.figure()
        img = mpimg.imread(str(self.src_name))

        cid = fig.canvas.mpl_connect('button_press_event', self._onclick)
        imgplot = plt.imshow(img)
        plt.show()

        draw_button = widgets.Button(description='Finish drawing polygon')
        dump_button = widgets.Button(description='Dump all polygons')
        draw_button.on_click(self._on_draw_button_clicked)
        dump_button.on_click(self._on_dump_button_clicked)
        return widgets.VBox([draw_button, dump_button])
