import os
import argparse
import shutil

import torch
import torch.multiprocessing as mp

from logger import logger, log_init
from train import start
import utils 
from config import cfg, cfg_init

def set_gpus(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        gpus = ','.join([str(g) for g in cfg.TRAIN.GPUS])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        logger.warning(f'WARNING, GPUS already set in env: CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES")}')
      
@utils.cfg_frz
def parallel_init():
    args = {}
    
    try:
        args = parse_args()
    except SystemExit as e:
        logger.warning('Couldnt trace args (run in jupyter?), starting local initialization')
        class args: local_rank=0
    cfg.PARALLEL.LOCAL_RANK = args.local_rank 
    cfg.PARALLEL.IS_MASTER = cfg.PARALLEL.LOCAL_RANK == 0 
        
    torch.cuda.set_device(cfg.PARALLEL.LOCAL_RANK)
    
    if cfg.PARALLEL.DDP: 
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cfg.PARALLEL.WORLD_SIZE = torch.distributed.get_world_size()
    else:
        cfg.PARALLEL.WORLD_SIZE = 1
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

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
    set_gpus(cfg)
    parallel_init()
    if cfg.PARALLEL.IS_MASTER: logger.debug('\n' + cfg.dump(indent=4))
    output_folder = init_output_folder(cfg)
    log_init(output_folder)
    start(cfg, output_folder, n_epochs=cfg.TRAIN.EPOCH_COUNT, use_cuda=True)
