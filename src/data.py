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


from _data import *
#from _data import ImageDataset, PairDataset, ConcatDataset, expander, expander_float, mean_std_dataset, create_transforms, TransformDataset, update_mean_std


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
        "train512": SegmentDataset(DATA_DIR/'split512_1/train/imgs', DATA_DIR/'split512_1/train/masks'),
        "val512": SegmentDataset(DATA_DIR/'split512_1/val/imgs', DATA_DIR/'split512_1/val/masks'),
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

def build_datasets(cfg, mode_train=True, num_proc=4):
    def train_trans_get(*args, **kwargs): return augs.get_aug(*args, **kwargs)
    transform_factory = {
            'TRAIN':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'VALID':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'TEST':{'factory':TransformDataset, 'transform_getter':train_trans_get},
        }

    extend_factories = {
             'PRELOAD':partial(PreloadingDataset, num_proc=num_proc, progress=tqdm),
    }
 
    datasets = create_datasets(cfg, init_datasets(cfg))
    mean, std = mean_std_dataset(datasets['TRAIN'])
    if cfg.TRANSFORMERS.STD == (0,) and cfg.TRANSFORMERS.MEAN == (0,):
        update_mean_std(cfg, mean, std)
    datasets = create_extensions(cfg, datasets, extend_factories)

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

def build_dataloaders(cfg, datasets, samplers=None, drop_last=False, pin=False):
    dls = {}
    batch_sizes = {'TRAIN': cfg.TRAIN.BATCH_SIZE,
                    'VALID': cfg.VALID.BATCH_SIZE}

    samplers = {'TRAIN':None, 'VALID':None}
    num_workers = cfg.TRAIN.NUM_WORKERS
    for kind, dataset in datasets.items():
        if cfg.PARALLEL.DDP and kind == 'TRAIN':
                samplers['TRAIN'] = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=cfg.PARALLEL.WORLD_SIZE,
                                                                    rank=cfg.PARALLEL.LOCAL_RANK, shuffle=True)
        sampler = samplers[kind]    
        shuffle = kind == 'TRAIN' if sampler is None else False
        batch_size = batch_sizes[kind] if batch_sizes[kind] is not None else 1
        dls[kind] = create_dataloader(dataset, sampler, shuffle, batch_size, num_workers, drop_last, pin)
            
    return dls




