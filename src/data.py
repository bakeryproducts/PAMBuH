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
from nn_sampler import SelectiveSampler, DistributedSamplerWrapper


class WeightedDataset:
    def __init__(self, dataset, scores, replacement=True):
        assert len(dataset) == len(scores), (len(dataset), len(scores))
        scores = (1 + scores) ** (2)
        self.scores = scores / scores.sum()
        self.dataset = dataset
        self.replacement = replacement
        self.idxs = list(range(len(self.dataset)))
        #self.sampler = WeightedRandomSampler(self.scores, len(self.scores), replacement=replacement)
    
    def __getitem__(self, _):
        num_samples = 1
        idx = np.random.choice(self.idxs, num_samples, self.replacement, self.scores)
        #idxs = list(self.sampler)
        return self.dataset[idx[0]]
        
    def __len__(self): return len(self.dataset)

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
    
class SSLDataset:
    def __init__(self, root, crop_size):
        self.dataset = ImageDataset(root, '*/*.png')
        self.dataset.process_item = expander
        self.dataset = MultiplyDataset(self.dataset, 4)
        self.d4 = albu.Compose([
                                albu.ShiftScaleRotate(0.0625, 0.2, 45, p=.2), 
                                albu.RandomCrop(*crop_size),
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
    def __init__(self, imgs_path, masks_path, mode_train=True, hard_mult=None, weights=None):
        self.img_folders = utils.get_filenames(imgs_path, '*', lambda x: False)
        self.masks_folders = utils.get_filenames(masks_path, '*', lambda x: False)
        self.mode_train = mode_train
        self.img_domains = {  
                         '4ef6695ce': 1,
                         'b9a3865fc': 0,
                         'e79de561c': 1,
                         '8242609fa': 0,
                         'cb2d976f4': 0,
                         '26dc41664': 1,
                         'b2dc8411c': 0,
                         'afa5e8098': 1,
                         '0486052bb': 0,
                         '1e2425f28': 1,
                         'c68fe75ea': 1,
                         'aaa6a05cc': 0,
                         '54f2eec69': 1,
                         '095bf7a1f': 1,
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

            #if weights: dataset = WeightedDataset(dataset, scores[imgf.stem])
            if hard_mult and self.img_domains[imgf.name]: dataset = MultiplyDataset(dataset, hard_mult)
            dss.append(dataset)
        
        self.dataset = ConcatDataset(dss)
    
    def _load_scores(self, path):
        with open(path, 'rb') as f:
            scores = pickle.load(f)
        return scores

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
    mult = cfg['TRAIN']['HARD_MULT']
    weights = cfg['TRAIN']['WEIGHTS']
    AuxDataset = partial(SegmentDataset, hard_mult=mult, weights=weights)
    SslDS = partial(SSLDataset, crop_size=cfg['TRANSFORMERS']['CROP'])
    
    DATASETS = {
        "train2048full": AuxDataset(DATA_DIR/'CUTS/cuts2048x25/imgs', DATA_DIR/'CUTS/cuts2048x25/masks'),
        "train2048x25": AuxDataset(DATA_DIR/'SPLITS/split2048x25/train/imgs', DATA_DIR/'SPLITS/split2048x25/train/masks'),
        "train2i": AuxDataset(DATA_DIR/'SPLITS/split2048i2/train/imgs', DATA_DIR/'SPLITS/split2048i2/train/masks'),

        "val2048x25": SegmentDataset(DATA_DIR/'SPLITS/split2048x25/val/imgs', DATA_DIR/'SPLITS/split2048x25/val/masks'),
        "val2i": SegmentDataset(DATA_DIR/'SPLITS/split2048i2/val/imgs', DATA_DIR/'SPLITS/split2048i2/val/masks'),

        "ssl_test": SslDS(DATA_DIR/'ssl_cortex'),
        "ssl_val": SslDS(DATA_DIR/'SPLITS/split2048i2/val/imgs'),

        #"backs_cort": AuxDataset(DATA_DIR/'backs_010_x25_cortex/imgs', DATA_DIR/'backs_010_x25_cortex/masks'),
        #"backs_rand": AuxDataset(DATA_DIR/'backs_020_x25_random/imgs', DATA_DIR/'backs_020_x25_random/masks'),

        #"backs_rand": SegmentDataset(DATA_DIR/'backs_x25_random/imgs', DATA_DIR/'backs_x25_random/masks'),
        #"backs_cort": AuxDataset(DATA_DIR/'backs_x25_cortex/imgs', DATA_DIR/'backs_x25_cortex/masks'),
        #"backs_medu": SegmentDataset(DATA_DIR/'backs_x25_medula/imgs', DATA_DIR/'backs_x25_medula/masks'),

        "backs_cort2i": SegmentDataset(DATA_DIR/'backs_x25_cortex_2i/imgs', DATA_DIR/'backs_x25_cortex_2i/masks'),
        "backs_rand2i": SegmentDataset(DATA_DIR/'backs_x25_random_2i/imgs', DATA_DIR/'backs_x25_random_2i/masks'),
        "backs_medu2i": SegmentDataset(DATA_DIR/'backs_x25_medula_2i/imgs', DATA_DIR/'backs_x25_medula_2i/masks'),


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
            'SSL':{'factory':TransformSSLDataset, 'transform_getter':train_trans_get},
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
             'CACHING':CachingDataset,
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
    drop_last = True
    sampler = None 

    if selective:
        weights = torch.ones(len(dataset))
        sampler = SelectiveSampler(weights) 

    if cfg.PARALLEL.DDP and mode == 'TRAIN':
        if sampler is None:
            sampler = DistributedSampler(dataset, num_replicas=cfg.PARALLEL.WORLD_SIZE, rank=cfg.PARALLEL.LOCAL_RANK, shuffle=True)

    shuffle = sampler is None

    dl = DataLoader(
        dataset,
        batch_size=cfg[mode]['BATCH_SIZE'],
        shuffle=shuffle,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=None,
        sampler=sampler,)
    return dl




