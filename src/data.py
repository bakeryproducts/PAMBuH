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
import albumentations as albu

import utils
import augs
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


class TransformDataset:
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = albu.Compose([]) if transforms is None else transforms
    
    def __getitem__(self, idx):
        img, mask = self.dataset.__getitem__(idx)
        augmented = self.transforms(image=img, mask=mask)
        return augmented["image"], augmented["mask"]
    
    def __len__(self): return len(self.dataset)

def create_transforms(cfg,
                      transform_factories,
                      dataset_types=['TRAIN', 'VALID', 'TEST']):
    transformers = {}
    for dataset_type in dataset_types:
        aug_type = cfg.TRANSFORMERS[dataset_type]['AUG']
        args={
            'aug_type':aug_type,
            'transforms_cfg':cfg.TRANSFORMERS
        }
        if transform_factories[dataset_type]['factory'] is not None:
            transform_getter = transform_factories[dataset_type]['transform_getter'](**args)
            transformer = partial(transform_factories[dataset_type]['factory'], transforms=transform_getter)
        else:
            transformer = lambda x: x
        transformers[dataset_type] = transformer
    return transformers    

def apply_transforms_datasets(datasets, transforms):
    return {dataset_type:transforms[dataset_type](dataset) for dataset_type, dataset in datasets.items()}


def expander(x):
    x = np.array(x)
    return x if len(x.shape) == 3 else np.repeat(np.expand_dims(x, axis=-1), 3, -1)

def expander_float(x):
    x = np.array(x).astype(np.float32) / 255.
    return x if len(x.shape) == 3 else np.repeat(np.expand_dims(x, axis=-1), 3, -1)

def update_mean_std(cfg, mean, std):
    was_frozen: False
    if cfg.is_frozen():
        cfg.defrost()
        was_frozen = True
        
    cfg.TRANSFORMERS.MEAN = tuple(mean.tolist())
    cfg.TRANSFORMERS.STD = tuple(std.tolist())
    if was_frozen: cfg.freeze()
        
def mean_std_dataset(dataset, parts=200):
    """
    taking every step-th image of dataset, averaging mean and std on them
    Image format is uint8, 0-255
    """
    mm, ss = [],[]
    step = max(1,len(dataset) // parts)
    for j, i in enumerate(dataset):
        if j % step == 0:
            d = i[0]/255.
            mm.append(d.mean(axis=(0,1)))
            ss.append(d.std(axis=(0,1)))
            #break
    return np.array(mm).mean(0), np.array(ss).mean(0)


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
                ids.process_item = expander
                mds.process_item = expander_float
                #ids.process_item = expander_float
                #mds.process_item = expander_float
            dss.append(PairDataset(ids, mds))
        
        self.dataset = ConcatDataset(dss)
    
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def _view(self, idx):
        pair = self.__getitem__(idx)
        return Image.blend(*pair,.5)
    

def init_datasets(cfg):
    DATA_DIR = Path(cfg.INPUTS).absolute()
    if not DATA_DIR.exists(): raise Exception(DATA_DIR)
    
    DATASETS = {
        "cuts512": SegmentDataset(DATA_DIR/'cuts512/imgs', DATA_DIR/'cuts512/masks'),
    }
    return  DATASETS

def create_datasets(cfg, all_datasets, dataset_types=['TRAIN', 'VALID', 'TEST']):
    converted_datasets = {}
    for dataset_type in dataset_types:
        data_field = cfg.DATA[dataset_type]
        datasets_strings = data_field.DATASETS
        if datasets_strings:
            datasets = [all_datasets[ds] for ds in datasets_strings]
            #ds = ConcatDataset(datasets) if len(datasets)>1 else datasets[0] 
            ds = datasets[0]
            converted_datasets[dataset_type] = ds
    return converted_datasets

def build_datasets(cfg, mode_train=True):
    def train_trans_get(*args, **kwargs): return augs.get_aug(*args, **kwargs)
    transform_factory = {
            'TRAIN':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'VALID':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'TEST':{'factory':TransformDataset, 'transform_getter':train_trans_get},
        }
    
    datasets = create_datasets(cfg, init_datasets(cfg))
    mean, std = mean_std_dataset(datasets['TRAIN'])
    if cfg.TRANSFORMERS.STD == (0,) and cfg.TRANSFORMERS.MEAN == (0,):
        update_mean_std(cfg, mean, std)

    transforms = create_transforms(cfg, transform_factory)
    datasets = apply_transforms_datasets(datasets, transforms)
    return datasets

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




