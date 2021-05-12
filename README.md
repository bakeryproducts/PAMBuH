# HUBMAP
kindney challenge


### We did this project locally, not in kaggle environment, so it can be resource heavy for that usage

Our rig was single node machine:
    -16CPU
    -4GPU V100 32gb
    -256gb RAM
    -a lot of memory, one folded model produce ~50gb of date at the end of inference phase


### Result of LB .925 / PB .949 can be achieved without ensembling with single 4fold model Unet-timm-regnety16 under 2hrs of training, default config
### Result of LB .926 / PB .950 was ensemble of 4 4folded models:
- Unet-regnety16
- Unet-regnetx32
- UnetPlusPlus-regnety16
- Unet-regnety16 with scse attention decoder

Training time for one model ~ 2hrs, 50-80 epochs of mixed data from grid sampling and object sampling
Inference time on kaggle was ~7hrs for all images, locally ~30min for 5 images



Data structure:
```
-src // soruce code
-input
    -HUBMAP // source folder extracted from zip file
    -bigmasks // tiff full-size masks; generated by data_gen.py
    -CUTS // tiles cut off from tiffs, paired with masks; data_gen.py
        -glomi_x33_1024
            -imgs
                -1e2425f28
                    0000001.png
                    0000002.png
                    ...
                -2f6ecfcdf
                    0000001.png
                    0000002.png
                    ...
                ...
            -masks
                ...
        -grid_x33_1024
            ...

    -SPLITS // train and val splits, we use 4-fold split, see split_gen.py; data_gen.py
        -glomi_split
            -0e
                -train
                    -imgs
                        -1e2425f28
                        ...
                    -masks
                        ...
                -val
                    -imgs
                        -2f6ecfcdf
                        ...
                    -masks
                        ...
            -2a
            -18
            -cc
        -grid_split
            -0e
            -2a
            -18
            -cc
            
-output // results folder
    -2021_feb_07_12_13_55 // trained model folder, IT MEANT TO BE ONE STRUCTURE, we are selecting this whole folder in inference module
        -model // model checkpoints, selected by best val / EMAval model score
            e100.pth
        -logs // tensorboard logs
        -src // copy of source code from root/src; for reproducing
        -cfg.yaml // model config
    ...
```


# Usage

### Install Requirements
One specific requirement is shallow library, src/shallow
```
pip3 install -r requirements.txt
cd src/shallow/ && pip3 install .

```
We are using rasterio as tiff reader and installing rasterio through pip should be enough.

But it uses GDAL under the hood, which can be pain to install, so just in case:
```
apt-get update
apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
gdal-config --version
pip install GDAL==2.2.3
```


### Create data

This line will read input/HUBMAP folder with source data and produce cutted tiles in 4 fold splits for training.
```
python3 src/data_gen.py

```
### Train model

Pytorch, albumentations, BCE+Dice, 4folds, Full precision

This will start training, default mode is 4 GPUS (we used V100 32gb)

ALl default params in src/configs/unet_gelb.yaml

Input data keys that connects `'train_0e_33'` to folder `input/SPLITS/glomi_split/0e/` in `data.py`

```
python3 src/train.py

```
This will create timestamped folder with results and logs in `output`


Single GPU usage: 
```yaml
PARALLEL:
  DDP: False
```

Single fold usage:
```yaml

DATA:
  TRAIN:
    DATASETS: ['train_0e_33',  'grid_0e_33']
    GPU_PRELOAD: False
    PRELOAD: False
    CACHE: False
    MULTIPLY: {'rate':1}
  VALID:
    FOLDS: ['val_0e_33' ]
    PRELOAD: False

```

### Inference

Inference working like this:
```
python3 src/run_inference.py TIMESTAMPED_FOLDER_WITH_RESULTS_FROM_TRAINING FOLDER_WITH_TEST_DATA
```
i.e.
```
python3 src/run_inference.py  2021_feb_07_12_13_55  input/HUBMAP/test
```

Script will select best model from checkpoints based on val score, and run inference on .tiff images in specified folder
There are switchable options in script, such as:

    -save RLE
    -do TTA
    -save Masks
    -multiprocessing with one image per GPU


This script is quite different from kaggle inference notebook, since we own and can use more resources locally

Gleb Sokolov, gleb.m.sokolov@gmail.com


