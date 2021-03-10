from operator import itemgetter
from random import choices, sample

import numpy as np
import torch
from torch.utils.data import DistributedSampler, Dataset
from torch.utils.data.sampler import BatchSampler, Sampler, WeightedRandomSampler

class DatasetFromSampler(Dataset):
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None: self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int: return len(self.sampler)
    
    
class DistributedSamplerWrapper(DistributedSampler):
    def __init__(self, sampler, num_replicas=None, rank=None, shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,)
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class SelectiveSampler(Sampler):
    def __init__(self, weights, num_draw=1.):
        if isinstance(num_draw, float): num_draw = int(num_draw * len(weights))
        self.sampler = WeightedRandomSampler(weights, num_draw , replacement=False)
        self.alpha = .95
                 
    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        return iter(self.dataset)

    def __len__(self): return len(self.sampler)
        
    def set_weights(self, ws): 
        for w, idx in zip(ws, self.dataset.sampler_list):
            self.sampler.weights[idx] = w*self.alpha + (1-self.alpha)*self.sampler.weights[idx]
    
    def _reorder(self, ws):
        return torch.tensor([x for _,x in sorted(zip(self.dataset.sampler_list, ws), reverse=False)], dtype=torch.float)

    def set_epoch(self, *args, **kwargs): pass


