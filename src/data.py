from PIL import Image
from pathlib import Path
from functools import partial, reduce
from collections import defaultdict
import multiprocessing as mp
from contextlib import contextmanager

import cv2
import torch
import numpy as np
from tqdm.auto import tqdm
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import ConcatDataset as TorchConcatDataset

import utils
import augs
from _data import *
from sampler import GdalSampler
from nn_sampler import SelectiveSampler, DistributedSamplerWrapper

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
                ids.process_item = expander
                mds.process_item = expander_float
            dss.append(PairDataset(ids, mds))
        
        self.dataset = ConcatDataset(dss)
    
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def _view(self, idx):
        pair = self.__getitem__(idx)
        return Image.blend(*pair,.5)

class TagSegmentDataset:
    def __init__(self, imgs_path, masks_path, mode_train=True):
        self.img_folders = utils.get_filenames(imgs_path, '*', lambda x: False)
        self.masks_folders = utils.get_filenames(masks_path, '*', lambda x: False)
        self.mode_train = mode_train
        img_domains = {  '0486052bb': 1,
                         '095bf7a1f': 0,
                         '1e2425f28': 0,
                         '2f6ecfcdf': 1,
                         '54f2eec69': 0,
                         'aaa6a05cc': 1,
                         'cb2d976f4': 1,
                         'e79de561c': 0}
        
        dss = []
        for imgf, maskf in zip(self.img_folders, self.masks_folders):
            ids = ImageDataset(imgf, '*.png')
            mds = TagImageDataset(maskf, '*.png', tags=img_domains)
            if self.mode_train:
                ids.process_item = expander
                mds.process_item = expander_float_item
            dss.append(PairDataset(ids, mds))
        
        self.dataset = ConcatDataset(dss)
    
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def _view(self, idx):
        img, mask, cl_id = self.__getitem__(idx)
        return Image.blend(img, mask,.5)
    

def init_datasets(cfg):
    """
        DATASETS dictionary:
            keys are custom names to use in unet.yaml
            values are actual Datasets on desired folders
    """
    DATA_DIR = Path(cfg.INPUTS).absolute()
    if not DATA_DIR.exists(): raise Exception(DATA_DIR)
    
    DATASETS = {
        "train1024x05": SegmentDataset(DATA_DIR/'SPLITS/split1024x05/train/imgs', DATA_DIR/'SPLITS/split1024x05/train/masks'),
        "val1024x05": SegmentDataset(DATA_DIR/'SPLITS/split1024x05/val/imgs', DATA_DIR/'SPLITS/split1024x05/val/masks'),
        "backs_30_1024": SegmentDataset(DATA_DIR/'backs030/imgs', DATA_DIR/'backs030/masks'),

        "full1024x25": SegmentDataset(DATA_DIR/'CUTS/cuts1024x25/imgs', DATA_DIR/'CUTS/cuts1024x25//masks'),
        "train1024x25": SegmentDataset(DATA_DIR/'SPLITS/split1024x25/train/imgs', DATA_DIR/'SPLITS/split1024x25/train/masks'),
        "val1024x25": SegmentDataset(DATA_DIR/'SPLITS/split1024x25/val/imgs', DATA_DIR/'SPLITS/split1024x25/val/masks'),
        "val2_1024x25": SegmentDataset(DATA_DIR/'SPLITS/split1024x25/val2/imgs', DATA_DIR/'SPLITS/split1024x25/val2/masks'),
        "backs_30_x25_1024": SegmentDataset(DATA_DIR/'backs030_x25/imgs', DATA_DIR/'backs030_x25/masks'),

        "train1024x25gray": SegmentDataset(DATA_DIR/'SPLITS/split1024x25_gray/train/imgs', DATA_DIR/'SPLITS/split1024x25_gray/train/masks'),
        "val1024x25gray": SegmentDataset(DATA_DIR/'SPLITS/split1024x25_gray/val/imgs', DATA_DIR/'SPLITS/split1024x25_gray/val/masks'),
        "backs_30_x25_1024gray": SegmentDataset(DATA_DIR/'backs030_x25_gray/imgs', DATA_DIR/'backs030_x25_gray/masks'),

        "train1024x25grayTTT": TagSegmentDataset(DATA_DIR/'SPLITS/split1024x25_gray/train/imgs', DATA_DIR/'SPLITS/split1024x25_gray/train/masks'),
        "val1024x25grayTTT": TagSegmentDataset(DATA_DIR/'SPLITS/split1024x25_gray/val/imgs', DATA_DIR/'SPLITS/split1024x25_gray/val/masks'),
        "backs_30_x25_1024grayTTT": TagSegmentDataset(DATA_DIR/'backs030_x25_gray/imgs', DATA_DIR/'backs030_x25_gray/masks'),
    }
    return  DATASETS

def create_datasets(cfg, all_datasets, dataset_types):
    converted_datasets = {}
    for dataset_type in dataset_types:
        data_field = cfg.DATA[dataset_type]
        datasets_strings = data_field.DATASETS
        if datasets_strings:
            datasets = [all_datasets[ds] for ds in datasets_strings]
            ds = TorchConcatDataset(datasets) if len(datasets)>1 else datasets[0] 
            converted_datasets[dataset_type] = ds
    return converted_datasets

def build_datasets(cfg, mode_train=True, num_proc=4, dataset_types=['TRAIN', 'VALID', 'TEST']):
    """
        Creates dictionary :
        {
            'TRAIN': <122254afsf9a>.obj.dataset,
            'VALID': <ascas924ja>.obj.dataset,
            'TEST': <das92hjasd>.obj.dataset,
        }

        train_dataset = build_datasets(cfg)['TRAIN']
        preprocessed_image, preprocessed_mask = train_dataset[0]

        All additional operations like preloading into memory, augmentations, etc
        is just another Dataset over existing one.
            preload_dataset = PreloadingDataset(MyDataset)
            augmented_dataset = TransformDataset(preload_dataset)
    """


    def train_trans_get(*args, **kwargs): return augs.get_aug(*args, **kwargs)
    transform_factory = {
            'TRAIN':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'VALID':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'VALID2':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'TEST':{'factory':TransformDataset, 'transform_getter':train_trans_get},
        }
    #tag_transform_factory = {
    #        'TRAIN':{'factory':TagTransformDataset, 'transform_getter':train_trans_get},
    #        'VALID':{'factory':TagTransformDataset, 'transform_getter':train_trans_get},
    #        'TEST':{'factory':TagTransformDataset, 'transform_getter':train_trans_get},
    #    }

    extend_factories = {
             'PRELOAD':partial(PreloadingDataset, num_proc=num_proc, progress=tqdm),
             'MULTIPLY':MultiplyDataset,
    }
 
    datasets = create_datasets(cfg, init_datasets(cfg), dataset_types)
    mean, std = mean_std_dataset(datasets['TRAIN'])
    if cfg.TRANSFORMERS.STD == (0,) and cfg.TRANSFORMERS.MEAN == (0,):
        update_mean_std(cfg, mean, std)

    datasets = create_extensions(cfg, datasets, extend_factories)
    transforms = create_transforms(cfg, transform_factory, dataset_types)
    datasets = apply_transforms_datasets(datasets, transforms)
    return datasets

def build_dataloaders(cfg, datasets, selective=True):
    '''
        Builds dataloader from datasets dictionary {'TRAIN':ds1, 'VALID':ds2}
        dataloaders :
            {
                'TRAIN': <sadd21e>.obj.dataloader,
                    ...
            }
    '''
    dls = {}
    for kind, dataset in datasets.items():
        dls[kind] = build_dataloader(cfg, dataset, kind, selective=selective)
    return dls


def build_dataloader(cfg, dataset, mode, selective):
    drop_last = False

    bs = cfg[mode]['BATCH_SIZE']
    num_workers = cfg.TRAIN.NUM_WORKERS
    
    sampler = None 
    if selective:
        _s = DistributedSampler(dataset, num_replicas=cfg.PARALLEL.WORLD_SIZE, rank=cfg.PARALLEL.LOCAL_RANK, shuffle=True)
        indxs = list(_s)
        dataset = FoldDataset(dataset, indxs)
        weights = torch.ones(len(dataset))
        #weights[:int(len(dataset)*.2)] *= 100 
        sampler = SelectiveSampler(weights) 

    if cfg.PARALLEL.DDP and mode == 'TRAIN':
        if sampler is None:
            sampler = DistributedSampler(dataset, num_replicas=cfg.PARALLEL.WORLD_SIZE, rank=cfg.PARALLEL.LOCAL_RANK, shuffle=True)

    shuffle = sampler is None

    dl = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=None,
        sampler=sampler,)
    return dl




