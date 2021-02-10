# PAMBuH
yendik challenge

# TODO

# SUBMIT!

1. Preload, precut, presave all data.
Data structure:
-input
    -train
        -imgs
            -04asd92
                0000001.png
                0000002.png
                ...
            -09asd2a
                0000001.png
                0000002.png
                ...
        -masks
            -04asd92
                0000001.png
                0000002.png
                ...
            -09asd2a
                0000001.png
                0000002.png
                ...
-output
    -2021_feb_07_12_13_55
        -model
            e100.pth
        -logs
        ...

2. PostProcess
_______________
Module
postprocess.py
from postprocess import postrocess

postprocess(infer_func, src_folder, dst_folder, save_predicts=True)
    '''
        src_folder - folder with test images
        dst_folder - folder to save output (RLE and predictions)

    '''

for image in test_images:
    mirror_pad(image)
    for block in padded_image:
        prediction = infer_func(block)
        valid_part = crop_center(prediction)
        result_image.insert(valid_part)

    save_raster
    save_rle

_____________
3. Submit 
???

4. pip package
5. Actual research: 
    cross-validate predictions on test data 
    put each fold on kaggle
    find worst test image
6. COde based:
    - DICE
    - load model
    - validation TB fix 
    - train blows out
    - Docker
    - 



n. Pipeline
```python
datasets = build_datasets()
dataloaders = build_dataloaders()
model = build_model()
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

