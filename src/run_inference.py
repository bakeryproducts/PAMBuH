import os
import time
from pathlib import Path
from logger import logger
from itertools import cycle
from functools import partial
import multiprocessing as mp
import multiprocessing.pool

from tqdm import tqdm

from postp import start, dump_to_csv, mp_func_wrapper

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess

if __name__ == '__main__':
    """
        MP GPU inference

    """
    #model_folder = 'output/925/'
    model_folder = 'output/2021_Mar_31_22_57_28_PAMBUH/'
    gpu_list = [0,1,2,3]
    threshold = int(.35 * 255)
    num_processes = len(gpu_list)

    img_names = list(Path('input/hm/test').glob('*.tiff'))
    img_names = [img_names[i] for i in [2,0,4,1,3]]#aa first
    #img_names = img_names[:2]

    save_predicts=True
    use_tta=True
    to_rle=True

    results = start_inf(model_folder,
                        img_names,
                        gpu_list,
                        threshold,
                        num_processes,
                        save_predicts,
                        use_tta,
                        to_rle)

    dump_to_csv(results, model_folder, threshold)


def start_inf(model_folder, img_names, gpu_list, threshold, num_processes, save_predicts, use_tta, to_rle):
    logger.log('DEBUG', '\n'.join(list([str(i) for i in img_names])))

    m = mp.Manager()
    results = m.dict()
    gpus = m.dict()
    for i in gpu_list:
        gpus[i] = False

    starter = partial(start, 
                        model_folder=model_folder,
                        threshold=threshold,
                        gpus=gpus,
                        results=results,
                        save_predicts=save_predicts,
                        use_tta=use_tta,
                        to_rle=to_rle)

    starter = partial(mp_func_wrapper, starter)
    args = [(name,) for name in img_names]

    with MyPool(num_processes) as p:    
        g = tqdm(p.imap(starter, args), total=len(img_names))
        for _ in g:
            pass

    return results 
