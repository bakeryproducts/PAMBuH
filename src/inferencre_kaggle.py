import warnings
from pathlib import Path
from collections import OrderedDict
import cv2
import torch
import numpy as np
import rasterio as rio
import albumentations as albu
import albumentations.pytorch as albu_pt
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
import segmentation_models_pytorch as smp
import ttach as tta


def load_model(model, model_folder_path):
    model = _load_model_state(model, model_folder_path)
    model.eval()
    return model


def _load_model_state(model, path):
    path = Path(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))['model_state']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            k = k.lstrip('module')[1:]
        new_state_dict[k] = v
    del state_dict
    model.load_state_dict(new_state_dict)
    del new_state_dict
    return model


def rescale(batch_img, scale):
    return torch.nn.functional.interpolate(batch_img, scale_factor=(scale, scale), mode='bilinear', align_corners=False)


class MegaModel():
    def __init__(self, model_folders, model_types, use_tta=True, use_cuda=True, threshold=.5, half_mode=False):
        self.use_cuda = use_cuda
        self.threshold = threshold
        self.use_tta = use_tta
        self.half_mode = half_mode
        self.averaging = 'mean'

        self._model_folders = model_folders  # list(Path(root).glob('*')) # root / model1 ; model2; model3; ...
        self._model_types = model_types
        self._model_scales = [3, 3, 3, 3]
        assert len(self._model_types) == len(self._model_folders)
        assert len(self._model_scales) == len(self._model_folders)

        self.models = self.models_init()
        mean, std = [0.6226, 0.4284, 0.6705], [0.1246, 0.1719, 0.0956]
        self.itransform = albu.Compose([albu.Normalize(mean=mean, std=std), ToTensor()])
        # print(self.models)

    def models_init(self):
        models = {}
        tta_transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Rotate90(angles=[0, 90])

            ]
        )
        for mt, mf, ms in zip(self._model_types, self._model_folders, self._model_scales):
            mf = Path(mf)
            model_names = list(mf.rglob('*.pth'))
            mm = []
            for mn in model_names:
                m = load_model(mt, mn)
                # print(mn)
                if self.use_tta: m = tta.SegmentationTTAWrapper(m, tta_transforms, merge_mode='mean')
                if self.use_cuda: m = m.cuda()
                if self.half_mode: m = m.half()
                mm.append(m)
            models[mf] = {'scale': ms, 'models': mm}
        return models

    def _prepr(self, img, scale):
        ch, H, W, dtype = *img.shape, img.dtype
        assert ch == 3 and dtype == np.uint8
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
        return self.itransform(image=img)['image']

    def each_forward(self, model, scale, batch):
        # print(batch.shape, batch.dtype, batch.max(), batch.min())
        with torch.no_grad():
            res = model(batch)
            res = torch.sigmoid(res)
        res = rescale(res, scale)
        return res

    def __call__(self, imgs, cuda=True):
        batch = None
        scale = None
        preds = []

        for mod_name, params in self.models.items():
            _scale = params['scale']
            if batch is None or scale != _scale:
                scale = _scale
                batch = [self._prepr(i, scale) for i in imgs]
                batch = torch.stack(batch, axis=0)
                if self.half_mode: batch = batch.half()
                if self.use_cuda: batch = batch.cuda()
            # print(batch.shape, batch.dtype, batch.max())
            models = params['models']
            _predicts = torch.stack([self.each_forward(m, scale, batch) for m in models])
            preds.append(_predicts)

        res = torch.vstack(preds).mean(0)
        res = res > self.threshold
        return res


