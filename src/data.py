from PIL import Image
from pathlib import Path
from functools import partial, reduce
from collections import defaultdict
import multiprocessing as mp
from contextlib import contextmanager

import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

import utils
from sampler import GdalSampler


class Dataset:
    def __init__(self, root, pattern):
        self.root = Path(root)
        self.pattern = pattern
        self.files = sorted(list(self.root.glob(self.pattern)))
        self._is_empty('There is no matching files!')
        
    def apply_filter(self, filter_fn):
        self.files = filter_fn(self.files)
        self._is_empty()

    def _is_empty(self, msg='There is no item in dataset!'): assert len(self.files) > 0
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return self.process_item(self.load_item(idx))
    def load_item(self, idx): raise NotImplementedError
    def process_item(self, item): return item
    

class ImageDataset(Dataset):
    def load_item(self, idx):
        img_path = self.files[idx]
        img = Image.open(str(img_path))
        return img
    
class PairDataset:
    def __init__(self, ds1, ds2):
        self.ds1, self.ds2 = ds1, ds2
        self.check_len()
    
    def __len__(self): return len(self.ds1)
    def check_len(self): assert len(self.ds1) == len(self.ds2)
    
    def __getitem__(self, idx):
        return self.ds1.__getitem__(idx), self.ds2.__getitem__(idx) 


class ConcatDataset:
    """
    To avoid recursive calls (like in torchvision variant)
    """
    def __init__(self, dss):
        self.length = 0
        self.ds_map = {}
        for i, ds in enumerate(dss):
            for j in range(len(ds)):
                self.ds_map[j+self.length] = i, self.length
            self.length += len(ds)
        self.dss = dss
    
    def load_item(self, idx):
        if idx >= self.__len__(): raise StopIteration
        ds_idx, local_idx = self.ds_map[idx]
        return self.dss[ds_idx].__getitem__(idx - local_idx)
    
    def _is_empty(self, msg='There is no item in dataset!'): assert len(self.files) > 0
    def __len__(self): return self.length
    def __getitem__(self, idx): return self.load_item(idx)


def expander(x):
    x = np.array(x)
    return x if len(x.shape) == 3 else np.repeat(np.expand_dims(x, axis=-1), 3, -1)

def expander_float(x):
    x = np.array(x).astype(np.float32) / 255.
    x = x.transpose(2,0,1)
    return x if len(x.shape) == 3 else np.repeat(np.expand_dims(x, axis=0), 3, 0)

class SegmentDataset:
    '''
    mode_train False  is for viewing imgs in ipunb
    imgs_path - path to folder with cutted samples: 'input/train/cuts512/'
    -train
        -cuts512
            -imgs
                -afe419239
                    0.png
                    1.png
            -masks
                -afe419239
                    0.png
                    1.png
    '''
    def __init__(self, imgs_path, masks_path, mode_train=True):
        self.img_folders = utils.get_filenames(imgs_path, '*', lambda x: False)
        self.masks_folders = utils.get_filenames(masks_path, '*', lambda x: False)
        self.mode_train = mode_train
        
        dss = []
        for imgf, maskf in zip(self.img_folders, self.masks_folders):
            ids = ImageDataset(imgf, '*.png')
            mds = ImageDataset(maskf, '*.png')
            if self.mode_train:
                # TODO CHANGE THIS BACK to expander, AUG DATASETS
                ids.process_item = expander_float
                mds.process_item = expander_float
            dss.append(PairDataset(ids, mds))
        
        self.dataset = ConcatDataset(dss)
    
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def _view(self, idx):
        pair = self.__getitem__(idx)
        return Image.blend(*pair,.5)
    
def build_datasets(cfg=None, mode_train=True):
    '''
    todo path from cfg
    '''
    root = Path('input/train/512')
    sd = SegmentDataset(root / 'imgs', root / 'masks', mode_train=mode_train)
    return {'TRAIN':sd}



def create_dataloader(dataset, sampler, shuffle, batch_size, num_workers, drop_last, pin):
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=None,
        sampler=sampler,
    )
    return dl

def build_dataloaders(datasets, samplers=None, batch_sizes=None, num_workers=1, drop_last=False, pin=False):
    dls = {}
    for kind, dataset in datasets.items():
        sampler = samplers[kind]    
        shuffle = kind == 'TRAIN' if sampler is None else False
        batch_size = batch_sizes[kind] if batch_sizes[kind] is not None else 1
        dls[kind] = create_dataloader(dataset, sampler, shuffle, batch_size, num_workers, drop_last, pin)
            
    return dls



