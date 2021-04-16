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

2. AUGs

3. 



5. COde based:
    - scale problem , .65/.5
    - 512?
    - exclude afa from valid total?
    - multiply hard even more
    - Smooth pseudo label, mix with hard
    - threshold ? Multifold??
    - pseudo on test
    - aug even harder for ssl
    - clean images research
    - tricky image research
    - aux classifier
    - mix of fold models
    - optimizerS schedule
    - join predicts on logits
    - 


6. pip package?



Papers:
Unet, resnet
- https://arxiv.org/pdf/2001.05548v1.pdf
- https://arxiv.org/pdf/1804.03999.pdf
- https://arxiv.org/pdf/1903.02740v1.pdf

