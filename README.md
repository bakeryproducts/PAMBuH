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
 - cut test image on grid
 - predict on Patches
 - cut test image on overlapping grid
 - predict on Patches
 - join predictions
 - RLE

3. Submit 
???

4. pip package
5. Actual research: 
    cross-validate predictions on test data 
    put each fold on kaggle
    find worst test image



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

