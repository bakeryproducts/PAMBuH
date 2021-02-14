import sys
from pathlib import Path
from PIL import Image
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as albu


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
        #print(m, epoch)
        
        res.append([i,epoch])
    idx = sorted(res, key=lambda x: -x[1])[0][0]
    return models[idx]

def rescale(batch_img, scale): return F.interpolate(batch_img, scale_factor=(scale, scale))

def preprocess_image(img, transform):
    # TODO scaling param
    DOWNSCALE = 2
    ch, H,W, dtype = *img.shape, img.dtype
    assert ch==3 and dtype==np.uint8
    img = img.transpose(1,2,0)

    img = cv2.resize(img, (W // DOWNSCALE, H // DOWNSCALE))
    return transform(image=img)['image']

def _infer_func(imgs, transform, model):
    batch = [preprocess_image(i, transform) for i in imgs]
    batch = torch.stack(batch,axis=0).cuda()
    #print(batch.shape, batch.dtype)
    with torch.no_grad():
        res = torch.sigmoid(model(batch))

    UPSCALE = 2
    res = rescale(res, UPSCALE)
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
    #sys.path.append(str(root) + 'src')
    import augs
    #import model as nn_model
    from config import cfg, cfg_init

    cfg_init(str(root / 'cfg.yaml'))
    cfg['PARALLEL']['DDP'] = False
    cfg['DATA']['TRAIN']['PRELOAD'] = False

    train_trans = augs.get_aug('light', cfg.TRANSFORMERS)
    transform = train_trans.transforms.transforms[-1] # norm and chw

    model_path = get_last_model(root / 'models')
    model = nn_model.load_model(cfg, str(model_path))
    model = model.cuda()

    return partial(_infer_func, transform=transform, model=model)




