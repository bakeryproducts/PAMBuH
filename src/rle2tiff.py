import pandas as pd
import numpy as np
import tifffile

dir_input = 'F:/kaggle/HuBMAP'
dir_output = 'F:/kaggle/HuBMAP/masks'


def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.bool)  # 0 - no glom
    for lo, hi in zip(starts, ends):
        img[lo:hi] = True      # 1 - glom
    return img.reshape(shape).T


def mask2tiff(tiff_name, dir_input, dir_output):      # tiff_name = '1e2425f28'
    train = pd.read_csv(f"{dir_input}/train.csv")
    index_rle = int(train[train['id'] == tiff_name].index.values)
    info = pd.read_csv(f"{dir_input}/HuBMAP-20-dataset_information.csv")
    index_csv = info[info['image_file'] == tiff_name + '.tiff'].index.values
    width = int(info.iloc[index_csv, 1])
    height = int(info.iloc[index_csv, 2])
    mask = rle2mask(train.iloc[index_rle, 1], (width, height))
    tifffile.imwrite(f"{dir_output}/{train.iloc[index_rle,0]}_mask.tiff", mask, photometric='minisblack')


tiff_name = '0486052bb'
mask2tiff(tiff_name, dir_input, dir_output)
