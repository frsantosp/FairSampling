import time
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count

from parameter_parser import parameter_parser
from utils import tab_printer, print_results
from sampler_targeted import Sampler
from tqdm import tqdm
import numpy as np

def run_sampler(arg):
    if arg[1]%64==0:
        print(arg[1],end = ' ', flush=True)
    nodes = arg[0].run_sample(arg[1])
    arg[0].write_sample( nodes, arg[1])

def main():
    args = parameter_parser()
    tab_printer(args)
    results = list()
    my_sampler = Sampler(args, print_ = False)
    print(f'starting computations on {cpu_count()} cores.')
    for p in range(3):       
        my_sampler.args.sample_size = 100*2**p
        if args.ego_exist == 1: 
            print(' run sampling number:', end = ' ')
            values = list()
            for i in range(args.sample_number):
                values.append((my_sampler ,i ))
            values = tuple(values)
            with Pool() as pool:
                results = pool.map(run_sampler, values)
            print('\n sampling process is over for samples of size:', 100*2**p)
        else:
            if args.sampler == 'GF':
                nodes = my_sampler.run_sample(0)
                my_sampler.write_sample( nodes, 0)
            else:
                for i in tqdm(range(args.sample_number)):
                    nodes = my_sampler.run_sample(i)
                    my_sampler.write_sample( nodes, i)
                print('Sampling process is over for samples of size:', 100*2**p)
if __name__ == '__main__':
    main()