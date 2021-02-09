from PIL import Image
from pathlib import Path
from functools import partial, reduce
from collections import defaultdict

import numpy as np


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
        amask = augmented["mask"][0]# as binary
        amask = amask.view(1,*amask.shape)

        return augmented["image"], amask
    
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
    for j in range(len(dataset)):
        if j % step == 0:
            i = dataset[j]
            d = i[0]/255.
            mm.append(d.mean(axis=(0,1)))
            ss.append(d.std(axis=(0,1)))
            #break
    return np.array(mm).mean(0), np.array(ss).mean(0)



