# PAMBuH
yendik challenge

Data structure:
```
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
```

# TODO

1. Kgl inference
    - analyze test images, reading speed MPxls / sec 
    - run some boilerplate code with import pseudo-model from external source
    - prototype MP loading single GPU inference pipeline (aka postp.py)
    - make sure that we are reading SUBDATASETS correctly. [3,H,W]

### KGL Pipeline 

    KEEP IT SIMPLE

#### Single process variant

    mega_model = MegaModel(dir, threshold=0.3, averaging_mode='mean', preprocess_fn=preprocess_fn) 
    # somthing like 16 models, each of size 200<s<400mb, all in GPU memory!

    images_names = get_names()

    scale = 3
    size, overlap = 512*3,256*3 # resulting size of crop should be dividable by scale
 
    rles = {}
    for i_fn in images_names:
        tile_reader = TileReader(i_fn, tile_size=size, overlap=overlap)
        predict = np.zeros(H, W) # binary
        batch = []
        for x,y,crop in tile_reader:
            if len(batch)  == 8:
                predict_tiles = mega_model.preprocess_and_infer(crop) # something like (BS, 1, CROP_H ,CROP_W), binary
                for x,y,tile_pred  in predict_tiles:
                    # careful on edges, every tile from predict of **fixed size**, i.e. 1,1536,1536
                    # maybe we should just stuck all predictions together and crop them to H, W size later
                    mask[x,y] = tile_pred
            else:
                batch.add([x,y,crop])
        if batch:
            # same prediction on non-full batch

        rles[i_fn] = do_rle(mask)
    
    df.save_from_rles(rles)
            
#### MP pipeline

    multiprocess readers, read and load tiles from image to Queue, 
    writer on main process checks Queue for tiles, loads them into batch, rest is in sync with SP pipeline


1. COde based:
    - threshold ? Multifold??
    - pseudo on test
    - mix of fold models
    - optimizerS schedule
    - SiLU
    - rg32

