import os.path
import shutil
import subprocess


def generate_sh():
    out_dir = 'sh'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cnt = 0
    alg_methods = ['proposed']      # 'baseline'
    update_methods = ["mean", "median", "trimmed_mean"]
    data_seeds = range(0, 100, 20)
    for alg_method in alg_methods:
        for update_method in update_methods:
            for data_seed in data_seeds:
                cnt += 1
                name = f"{alg_method}-{update_method}-{data_seed}"
                hh, mm = divmod(cnt, 60)
                s = fr"""#!/bin/bash

#SBATCH --job-name={name}         # create a short name for your job
#SBATCH --time=2:{mm}:00          # total run time limit (HH:MM:SS)\
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output={out_dir}/out_{name}.txt
#SBATCH --error={out_dir}/err_{name}.txt

module purge
cd /scratch/gpfs/ky8517/ifca/mnist
module load anaconda3/2023.3
conda activate ifca_torch

pwd
python3 -V
uname -a 
hostname -s
date
 
PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 train_cluster_mnist.py --alg_method {alg_method} --update_method {update_method} --data_seed {data_seed} &> {out_dir}/{name}.txt 

wait
date 
echo 'done'     
    """
                out_sh = f'{out_dir}/{name}.sh'
                # print(out_sh)
                with open(out_sh, 'w') as f:
                    f.write(s)

                cmd = f'sbatch {out_sh}'
                ret = subprocess.run(cmd, shell=True)
                print(cmd, ret)
            
    return cnt


if __name__ == '__main__':
    cnt = generate_sh()
    print(f'\n***total submitted jobs: {cnt}')
