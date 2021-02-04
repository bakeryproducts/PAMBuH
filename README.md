# PAMBuH
yendik challenge

## GDAL install, ubuntu / debian
```bash
apt-get update
apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
gdal-config --version
pip install GDAL==2.2.3
```

# TODO

1. Sampler 
```python
s = Sampler('img.tif', 'img.json', wh=(256,256), maskwh=(256,256))
img, mask = s[3]
```
2. Rle Code, with MINIMUM amount of bytes per pixel (idealy, 1bit, since mask is binary)
```python
rle2tiff('rle.json', 'img_rle.tiff')
```
3. Pipeline
```python
datasets = build_datasets()
dataloaders = build_dataloaders()
model = build_model()
start(model, dataloaders)
```

Papers:
Unet, resnet
- https://arxiv.org/pdf/2001.05548v1.pdf
- https://arxiv.org/pdf/1804.03999.pdf
- https://arxiv.org/pdf/1903.02740v1.pdf


### Full image prediction
TTA, d4 group
overlapping tiles
merging like [that](https://github.com/Vooban/Smoothly-Blend-Image-Patches)
someone shoud rewrite that for gpu, torch

