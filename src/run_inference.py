import multiprocessing as mp
import multiprocessing.pool
#mp.set_start_method('spawn')
import  time
import os
from pathlib import Path
from logger import logger
from functools import partial
from itertools import cycle

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
    #model_folder = 'output/925/'
    model_folder = 'output/2021_Mar_20_18_41_28_PAMBUH/'
    threshold = int(.4 * 255)#105

    img_names = Path('input/hm/test').glob('*.tiff')
    img_names = list(img_names)
    img_names = [img_names[i] for i in [2,0,4,1,3]]#aa first
    logger.log('DEBUG', list(img_names))

    #gpus = cycle([0,1,2,3])
    num_processes = 4
    #gpus = [0,1,2,3]

    m = mp.Manager()
    results = m.dict()
    gpus = m.dict()
    for i in range(4):
        gpus[i] = False

    starter = partial(start, model_folder=model_folder, threshold=threshold, gpus=gpus, results=results)
    starter = partial(mp_func_wrapper, starter)
    #args = [(name, next(gpus)) for name in img_names]
    args = [(name,) for name in img_names]
    #args = [(name, gpu) for name, gpu in zip(img_names, gpus)])]

    with MyPool(num_processes) as p:    
        g = tqdm(p.imap(starter, args), total=len(img_names))
        for _ in g:
            pass
    
    print(results.keys())
    dump_to_csv(results, model_folder, threshold)
