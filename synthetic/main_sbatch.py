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
    for n in [100]:
        for d in [20, 50, 100, 200, 500]:
            for update_method in ['mean', 'median', 'trimmed_mean']:
                for alg_method in ['proposed']:
                    cnt += 1
                    name = f"n_{n}-d_{d}-{update_method}-{alg_method}"
                    hh, mm = divmod(cnt, 60)
                    s = fr"""#!/bin/bash

#SBATCH --job-name={name}         # create a short name for your job
#SBATCH --time=48:{mm}:00          # total run time limit (HH:MM:SS)
#SBATCH --output={out_dir}/out_{name}.txt
#SBATCH --error={out_dir}/err_{name}.txt

module purge
cd /scratch/gpfs/ky8517/ifca/synthetic
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_ifca python=3.10.4
conda activate py3104_ifca

pwd
python3 -V
uname -a 
hostname -s

# PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 demo.py -n {n} -d {d} --update_method {update_method} --alg_method {alg_method}
# PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 demo.py -n {n} -d {d} --update_method {update_method} --alg_method {alg_method} > '{out_dir}/out_{name}.txt' 2>&1
PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 run_all.py -n {n} -d {d} --update_method {update_method} --alg_method {alg_method}
# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
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
