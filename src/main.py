import os
import argparse
import shutil

import torch
import torch.multiprocessing as mp

from logger import logger, log_init
from train import start
import utils 
from config import cfg, cfg_init


def init_output_folder(cfg):
    output_folder=None
    if cfg.PARALLEL.LOCAL_RANK == 0:
        output_folder = utils.make_folders(cfg)
        shutil.copytree('src', str(output_folder/'src'))
        #if os.path.exists('/mnt/src'): shutil.copytree('/mnt/src', str(output_folder/'src'))
        utils.dump_params(cfg, output_folder)
    return output_folder


if __name__  == "__main__":
    cfg_init('src/configs/unet.yaml')
    #utils.set_gpus(cfg)
    #utils.parallel_init()
    if cfg.PARALLEL.IS_MASTER: logger.debug('\n' + cfg.dump(indent=4))
    output_folder = init_output_folder(cfg)
    log_init(output_folder)
    
    start(cfg, output_folder, n_epochs=cfg.TRAIN.EPOCH_COUNT)
