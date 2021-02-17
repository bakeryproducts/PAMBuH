import sys
from pathlib import Path
from PIL import Image
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as albu

import augs

class CfgParse:
    def __init__(self, cfg_file):
        self.compose = albu.Compose
        data = self.parse_yaml(cfg_file)
        self.mean = data['TRANSFORMERS']['MEAN']
        self.std = data['TRANSFORMERS']['STD']        
        self.scale = self.get_scale(data)
        assert self.scale is not None

    def get_scale(self, data):
        return  data['TRAIN'].get('DOWNSCALE', None)

    def parse_yaml(self, cfg_file):
        with open(cfg_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return data

    def norm(self): 
        return self.compose([albu.Normalize(mean=self.mean, std=self.std), augs.ToTensor()])

def parse_model_path(p):
    name = str(p.name)
    epoch = name.split('_')[0]
    return int(epoch[1:])

def get_last_model(src):
    # assumes that model name is of type e500_blabla.pth, sorted by epoch #500
    models = list(Path(src).glob('*.pth'))
    res = []
    for i, m in enumerate(models):
        epoch = parse_model_path(m)
        res.append([i,epoch])
    idx = sorted(res, key=lambda x: -x[1])[0][0]
    return models[idx]

def rescale(batch_img, scale): return F.interpolate(batch_img, scale_factor=(scale, scale))

def preprocess_image(img, scale, transform):
    ch, H,W, dtype = *img.shape, img.dtype
    assert ch==3 and dtype==np.uint8
    img = img.transpose(1,2,0)

    img = cv2.resize(img, (W // scale, H // scale))
    return transform(image=img)['image']

def _infer_func(imgs, transform, scale, model):
    batch = [preprocess_image(i, scale, transform) for i in imgs]
    batch = torch.stack(batch,axis=0).cuda()
    #print(batch.shape, batch.dtype)
    with torch.no_grad():
        res = torch.sigmoid(model(batch))

    res = rescale(res, scale)
    return res
    
def get_infer_func(root):
    """
        Wraps model and preprocessing in one function with argument [img, ...]
        img is HWC (because of transforms)
        Returns torch tensor on CUDA, [len(inp), 1, H, W]
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("model", str(root)+'src/model.py')
    nn_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nn_model)

    root = Path(root)
    cfg_data = CfgParse(root/'cfg.yaml')
    transform = cfg_data.norm() 

    model_path = get_last_model(root / 'models')
    model = nn_model.load_model(_, str(model_path))
    model = model.cuda()

    return partial(_infer_func, transform=transform, scale=cfg_data.scale, model=model)


