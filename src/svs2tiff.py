import os
import numpy as np
import rasterio
from pathlib import Path
import argparse


def svs2tiff(path):
    svs_files = list((Path(path)).glob('*.svs'))
    for svs_file in svs_files:
        print(f"{svs_file}")
        img = rasterio.open(f"{svs_file}", 'r')
        img_array = img.read()
        h = img_array.shape[1]
        w = img_array.shape[2]
        print(w, h)
        svs_file = os.path.splitext(svs_file)[0]
        img_tiff = rasterio.open(f"{svs_file}.tiff", 'w', driver='GTiff', height=h, width=w, count=3, nbits=8, dtype=np.uint8)
        img_tiff.write(img_array)
        img_tiff.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVS path')
    parser.add_argument('path', type=str, help='SVS path')
    args = parser.parse_args()
    svs2tiff(args.path)
    print(args.path)


'''
if __name__ == '__main__':
    path = 'F:/kaggle/dsmatch/k7nvtgn2x6-3/DATASET_A_DIB'
    svs2tiff(path)

'''
