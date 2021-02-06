import pandas as pd
import numpy as np
import rasterio

dir_input = '../input/hm'
dir_output = '../input/bigmasks'

def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.bool) 
    for lo, hi in zip(starts, ends):
        img[lo:hi] = True
    return img.reshape(shape).T # .T ?? WTF

def mask2tiff(pic_num, dir_input, dir_output):
    ds = rasterio.open(f"{dir_input}/train/{train_df.iloc[pic_num,0]}.tiff")
    w,h = ds.shape
    ds.close()
    del ds
    mask = rle2mask(train_df.iloc[pic_num, 1], (h,w))
    dst = rasterio.open(f"{dir_output}/{train_df.iloc[pic_num,0]}_mask.tiff", 'w', driver='GTiff',
                            height = w, width = h, count=1, nbits=1, dtype=np.uint8)
    dst.write(mask.astype(np.uint8), 1)
    dst.close()
    del dst

train = pd.read_csv(f"{dir_input}/train.csv")
info = pd.read_csv(f"{dir_input}/HuBMAP-20-dataset_information.csv")
train_df = pd.read_csv(f"{dir_input}/train.csv")

for idx in range (8):
    print(idx)
    mask2tiff(idx, dir_input, dir_output)
