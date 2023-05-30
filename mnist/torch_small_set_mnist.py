
out_dir = "small_set-alpha_01-beta_01"

import torch.multiprocessing as mp
from train_cluster_mnist_small_set import main

if __name__ == '__main__':
    num_processes = 4

    alg_methods = ['proposed', 'baseline']
    update_methods = ["mean", "median", "trimmed_mean"]
    data_seeds = range(0, 100, 50)
    for alg_method in alg_methods:
        for update_method in update_methods:
            for data_seed in data_seeds:
                # cmd = "python3 -u train_cluster_mnist_small_set.py --data-seed ${data_seed} --update_method ${update_method} --alg_method ${alg_method}"
                # echo
                # "$cmd"
                # _out_dir = "${out_dir}/${alg_method}/${update_method}/${data_seed}"
                # if [ ! -d "$_out_dir"]; then
                # mkdir - p
                # "$_out_dir"

                processes = []
                for rank in range(num_processes):
                    p = mp.Process(target=main, args={})
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
