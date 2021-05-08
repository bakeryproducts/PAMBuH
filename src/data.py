import pickle
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


class TransformSSLDataset:
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = albu.Compose([]) if transforms is None else transforms
        self.norm = self.transforms[-1]
    
    def __getitem__(self, idx):
        i = self.dataset.__getitem__(idx)
        ai = self.transforms(image=i)['image']
        i = self.norm(image=i)['image']
        return ai, i
    
    def __len__(self): return len(self.dataset)
    
class TagDataset:
    def __init__(self, ds, tag): self.ds, self.tag = ds, tag
    def __getitem__(self, idx): return (*self.ds[idx], self.tag)
    def __len__(self): return len(self.ds)

class SSLDataset:
    def __init__(self, root, crop_size):
        self.dataset = ImageDataset(root, '*/*.png')
        self.dataset.process_item = expander
        self.dataset = MultiplyDataset(self.dataset, 8)
        self.d4 = albu.Compose([
                                albu.RandomCrop(*crop_size),
                                albu.ShiftScaleRotate(0.0625, 0.1, 45, p=.3), 
                                albu.Flip(),
                                albu.RandomRotate90()
                                ])
            
    def __getitem__(self, idx):
        i = self.dataset[idx]
        i = self.d4(image=i)['image']
        return i

    def __len__(self): return len(self.dataset)


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
    def __init__(self, root, mode_train=True, hard_mult=None, weights=None, frozen=False):
        imgs_path, masks_path = root / 'imgs', root / 'masks'
        self.img_folders = utils.get_filenames(imgs_path, '*', lambda x: False)
        self.masks_folders = utils.get_filenames(masks_path, '*', lambda x: False)
        self.mode_train = mode_train
        self.img_domains = {  
                         '4ef6695ce': 1,
                         'e79de561c': 1,
                         '26dc41664': 1,
                         'afa5e8098': 1,
                         '1e2425f28': 1,
                         'c68fe75ea': 1,
                         '54f2eec69': 1,
                         '095bf7a1f': 1,
                         'b9a3865fc': 0,
                         '8242609fa': 0,
                         'cb2d976f4': 0,
                         'b2dc8411c': 0,
                         '0486052bb': 0,
                         'aaa6a05cc': 0,
                         '2f6ecfcdf': 0,
                         }
        scores = self._load_scores(weights) if weights else None
        
        dss = []
        for imgf, maskf in zip(self.img_folders, self.masks_folders):
            ids = ImageDataset(imgf, '*.png')
            mds = ImageDataset(maskf, '*.png')
            if self.mode_train:
                ids.process_item = expander
                mds.process_item = expander_float
            dataset = PairDataset(ids, mds)

            if hard_mult and self.img_domains[imgf.name]: dataset = MultiplyDataset(dataset, hard_mult)
            if frozen:
                if not self.img_domains[imgf.name]: dss.append(dataset)
            else:
                dss.append(dataset)
        
        self.dataset = ConcatDataset(dss)
    
    def _load_scores(self, path):
        with open(path, 'rb') as f:
            scores = pickle.load(f)
        return scores

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def _view(self, idx):
        a,b = self.__getitem__(idx)
        if b.mode == 'L': b = b.convert('RGB')
        return Image.blend(a,b,.5)

class TagSegmentDataset:
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
    def __init__(self, root, mode_train=True, hard_mult=None, frozen=False):
        imgs_path, masks_path = root / 'imgs', root / 'masks'
        self.img_folders = utils.get_filenames(imgs_path, '*', lambda x: False)
        self.masks_folders = utils.get_filenames(masks_path, '*', lambda x: False)
        self.mode_train = mode_train
        self.img_domains = {  
                         '4ef6695ce': 1,
                         'e79de561c': 1,
                         '26dc41664': 1,
                         'afa5e8098': 1,
                         '1e2425f28': 1,
                         'c68fe75ea': 1,
                         '54f2eec69': 1,
                         '095bf7a1f': 1,
                         'b9a3865fc': 0,
                         '8242609fa': 0,
                         'cb2d976f4': 0,
                         'b2dc8411c': 0,
                         '0486052bb': 0,
                         'aaa6a05cc': 0,
                         '2f6ecfcdf': 0,
                         }
        
        dss = []
        for imgf, maskf in zip(self.img_folders, self.masks_folders):
            ids = ImageDataset(imgf, '*.png')
            mds = ImageDataset(maskf, '*.png')
            if self.mode_train:
                ids.process_item = expander
                mds.process_item = expander_float
            dataset = PairDataset(ids, mds)
            dataset = TagDataset(dataset, self.img_domains.get(imgf.name, 0))

            if hard_mult and self.img_domains[imgf.name]: dataset = MultiplyDataset(dataset, hard_mult)
            if frozen:
                if not self.img_domains[imgf.name]: dss.append(dataset)
            else: dss.append(dataset)
        
        self.dataset = ConcatDataset(dss)
    
    def _load_scores(self, path):
        with open(path, 'rb') as f:
            scores = pickle.load(f)
        return scores

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def _view(self, idx):
        a,b = self.__getitem__(idx)
        if b.mode == 'L': b = b.convert('RGB')
        return Image.blend(a,b,.5)


def init_datasets(cfg):
    """
        DATASETS dictionary:
            keys are custom names to use in unet.yaml
            values are actual Datasets on desired folders
    """
    DATA_DIR = Path(cfg.INPUTS).absolute()
    if not DATA_DIR.exists(): raise Exception(DATA_DIR)
    mult = cfg['TRAIN']['HARD_MULT']
    weights = cfg['TRAIN']['WEIGHTS']

    use_tag = False
    D = SegmentDataset if not use_tag else TagSegmentDataset

    AuxDataset = partial(D, hard_mult=mult)
    AuxFrozenDataset = partial(D, hard_mult=mult, frozen=True)
    AuxFPDataset = partial(D, hard_mult=10)
    AuxPseudoDataset = D

    SslDS = partial(SSLDataset, crop_size=cfg['TRANSFORMERS']['CROP'])
    
    DATASETS = {
        "sclero_33": D(DATA_DIR/'scleros_glomi/scle_cuts_x33_1024/'), 

        "grid_0e_33": AuxDataset(DATA_DIR/'backs/grid_splits_33_1024/0e/train/'),
        "grid_2a_33": AuxDataset(DATA_DIR/'backs/grid_splits_33_1024/2a/train/'),
        "grid_18_33": AuxDataset(DATA_DIR/'backs/grid_splits_33_1024/18/train/'),
        "grid_cc_33": AuxDataset(DATA_DIR/'backs/grid_splits_33_1024/cc/train/'),

        "val_0e_33": SegmentDataset(DATA_DIR/'SPLITS/f_split_33_1024/0e/val/'),
        "val_2a_33": SegmentDataset(DATA_DIR/'SPLITS/f_split_33_1024/2a/val/'),
        "val_18_33": SegmentDataset(DATA_DIR/'SPLITS/f_split_33_1024/18/val/'),
        "val_cc_33": SegmentDataset(DATA_DIR/'SPLITS/f_split_33_1024/cc/val/'),

        "train_0e_33": AuxDataset(DATA_DIR/'SPLITS/f_split_33_1024/0e/train/'),
        "train_2a_33": AuxDataset(DATA_DIR/'SPLITS/f_split_33_1024/2a/train/'),
        "train_18_33": AuxDataset(DATA_DIR/'SPLITS/f_split_33_1024/18/train/'),
        "train_cc_33": AuxDataset(DATA_DIR/'SPLITS/f_split_33_1024/cc/train/'),

        "grid_0e_33_B": AuxDataset(DATA_DIR/'backs/grid_splits_33_1024_B/0e/train/'),
        "grid_18_33_B": AuxDataset(DATA_DIR/'backs/grid_splits_33_1024_B/18/train/'),
        "grid_ab_33_B": AuxDataset(DATA_DIR/'backs/grid_splits_33_1024_B/ab/train/'),
        "grid_b2_33_B": AuxDataset(DATA_DIR/'backs/grid_splits_33_1024_B/b2/train/'),

        "val_0e_33_B": SegmentDataset(DATA_DIR/'SPLITS/f_split_33_1024_B/0e/val/'),
        "val_18_33_B": SegmentDataset(DATA_DIR/'SPLITS/f_split_33_1024_B/18/val/'),
        "val_ab_33_B": SegmentDataset(DATA_DIR/'SPLITS/f_split_33_1024_B/ab/val/'),
        "val_b2_33_B": SegmentDataset(DATA_DIR/'SPLITS/f_split_33_1024_B/b2/val/'),

        "train_0e_33_B": AuxDataset(DATA_DIR/'SPLITS/f_split_33_1024_B/0e/train/'),
        "train_18_33_B": AuxDataset(DATA_DIR/'SPLITS/f_split_33_1024_B/18/train/'),
        "train_ab_33_B": AuxDataset(DATA_DIR/'SPLITS/f_split_33_1024_B/ab/train/'),
        "train_b2_33_B": AuxDataset(DATA_DIR/'SPLITS/f_split_33_1024_B/b2/train/'),

        "pseudo_33": AuxPseudoDataset(DATA_DIR/'CUTS/pseudo_x33/'),
        "pseudo_33_b": AuxPseudoDataset(DATA_DIR/'CUTS/pseudo_x33_b/'),

# _______________________________________________________ 33________________

        #"train_0e_33": AuxDataset(DATA_DIR/'SPLITS/f_split_b_33/0e/train/'),
        "train_0e_33_frozen": AuxFrozenDataset(DATA_DIR/'SPLITS/f_split_b_33/0e/train/'),
        "train_0e_33_soft": AuxDataset(DATA_DIR/'SPLITS/f_split_soft_b_33/0e/train/'),
        #"val_0e_33": SegmentDataset(DATA_DIR/'SPLITS/f_split_b_33/0e/val/'),

        #"train_2a_33": AuxDataset(DATA_DIR/'SPLITS/f_split_b_33/2a/train/'),
        "train_2a_33_frozen": AuxFrozenDataset(DATA_DIR/'SPLITS/f_split_b_33/2a/train/'),
        "train_2a_33_soft": AuxDataset(DATA_DIR/'SPLITS/f_split_soft_b_33/2a/train/'),
        #"val_2a_33": SegmentDataset(DATA_DIR/'SPLITS/f_split_b_33/2a/val/'),

        #"train_18_33": AuxDataset(DATA_DIR/'SPLITS/f_split_b_33/18/train/'),
        "train_18_33_frozen": AuxFrozenDataset(DATA_DIR/'SPLITS/f_split_b_33/18/train/'),
        "train_18_33_soft": AuxDataset(DATA_DIR/'SPLITS/f_split_soft_b_33/18/train/'),
        #"val_18_33": SegmentDataset(DATA_DIR/'SPLITS/f_split_b_33/18/val/'),

        #"train_cc_33": AuxDataset(DATA_DIR/'SPLITS/f_split_b_33/cc/train/'),
        "train_cc_33_frozen": AuxFrozenDataset(DATA_DIR/'SPLITS/f_split_b_33/cc/train/'),
        "train_cc_33_soft": AuxDataset(DATA_DIR/'SPLITS/f_split_soft_b_33/cc/train/'),
        #"val_cc_33": SegmentDataset(DATA_DIR/'SPLITS/f_split_b_33/cc/val/'),


    }
    return  DATASETS

def create_datasets(cfg, all_datasets, dataset_types):
    converted_datasets = {}
    for dataset_type in dataset_types:
        data_field = cfg.DATA[dataset_type]
        if data_field.DATASETS != (0,):
            datasets = [all_datasets[ds] for ds in data_field.DATASETS]
            ds = TorchConcatDataset(datasets) if len(datasets)>1 else datasets[0] 
            converted_datasets[dataset_type] = ds
        elif data_field.FOLDS != (0,):
            __datasets = []
            for dss in data_field.FOLDS:
                datasets = [all_datasets[ds] for ds in dss]
                ds = TorchConcatDataset(datasets) if len(datasets)>1 else datasets[0] 
                __datasets.append(ds)

            converted_datasets[dataset_type] = __datasets

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
            #'TRAIN':{'factory':TagTransformDataset, 'transform_getter':train_trans_get},
            'TRAIN':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'VALID':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'VALID2':{'factory':TransformDataset, 'transform_getter':train_trans_get},
            'SSL':{'factory':TransformSSLDataset, 'transform_getter':train_trans_get},
            'TEST':{'factory':TransformDataset, 'transform_getter':train_trans_get},
        }

    extend_factories = {
             'PRELOAD':partial(PreloadingDataset, num_proc=num_proc, progress=tqdm),
             'MULTIPLY':MultiplyDataset,
             'CACHING':CachingDataset,
    }
 
    datasets = create_datasets(cfg, init_datasets(cfg), dataset_types)
    #TODO refact
    if isinstance(datasets['TRAIN'], list):
        # fold mode

        res_datasets = {'TRAIN':[], 'VALID':[]}
        for tds, vds in zip(datasets['TRAIN'], datasets['VALID']):
            datasets_fold = {'TRAIN': tds, 'VALID':vds}
            datasets_fold = create_extensions(cfg, datasets_fold, extend_factories)
            transforms = create_transforms(cfg, transform_factory, dataset_types)
            datasets_fold = apply_transforms_datasets(datasets_fold, transforms)
            res_datasets['TRAIN'].append(datasets_fold['TRAIN'])
            res_datasets['VALID'].append(datasets_fold['VALID'])
        return res_datasets

    else:
        mean, std = mean_std_dataset(datasets['TRAIN'])
        if cfg.TRANSFORMERS.STD == (0,) and cfg.TRANSFORMERS.MEAN == (0,):
            update_mean_std(cfg, mean, std)

        datasets = create_extensions(cfg, datasets, extend_factories)
        transforms = create_transforms(cfg, transform_factory, dataset_types)
        datasets = apply_transforms_datasets(datasets, transforms)
        return datasets

def build_dataloaders(cfg, datasets, selective=False):
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
    drop_last = True
    sampler = None 

    if cfg.PARALLEL.DDP and (mode == 'TRAIN' or mode == 'SSL'):
        if sampler is None:
            sampler = DistributedSampler(dataset, num_replicas=cfg.PARALLEL.WORLD_SIZE, rank=cfg.PARALLEL.LOCAL_RANK, shuffle=True)

    num_workers = cfg.TRAIN.NUM_WORKERS 
    #if mode == 'TRAIN': shuffle = sampler is None
    #else: shuffle = False
    shuffle = sampler is None

    dl = DataLoader(
        dataset,
        batch_size=cfg[mode]['BATCH_SIZE'],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=None,
        sampler=sampler,)
    return dl


