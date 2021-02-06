import pandas as pd
import numpy as np
import skimage.io
import tifffile

dir_input = 'F:/kaggle/HuBMAP'
dir_output = 'F:/kaggle/HuBMAP/masks'

def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8) # 0 - no glom
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1 # 0 - glom
    return img.reshape(shape).T

def mask2tiff(pic_num, dir_input, dir_output, rle2mask):
    image = skimage.io.imread(f"{dir_input}/train/{train_df.iloc[pic_num,0]}.tiff")
    mask = rle2mask(train_df.iloc[pic_num, 1], (image.shape[-2], image.shape[-3]))
    mask_norm = mask.astype(bool)
    tifffile.imwrite(f"{dir_output}/{train_df.iloc[pic_num,0]}_mask.tiff", mask_norm, photometric='minisblack')

train = pd.read_csv(f"{dir_input}/train.csv")
info = pd.read_csv(f"{dir_input}/HuBMAP-20-dataset_information.csv")
train_df = pd.read_csv(f"{dir_input}/train.csv")

for pic_num in range (8):                   # tiff output loop
    mask2tiff(pic_num, dir_input, dir_output, rle2mask)\