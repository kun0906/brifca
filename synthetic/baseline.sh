#!/bin/bash

#SBATCH --job-name=BS         # create a short name for your job
#SBATCH --nodes=3                # node count
#SBATCH --time=48:1:00          # total run time limit (HH:MM:SS)
#SBATCH --output=sh/BS.txt
#SBATCH --error=sh/BS.txt

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

# PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 demo.py -n 100 -d 20 --update_method mean --alg_method proposed
# PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 demo.py -n 100 -d 20 --update_method mean --alg_method proposed > 'sh/out_n_100-d_20-mean-proposed.txt' 2>&1
srun python3 run_all_plot.py -p 5 -m 200 --update_method trimmed_mean --alg_method baseline &> '5_200.txt' &
srun python3 run_all_plot.py -p 10 -m 400 --update_method trimmed_mean --alg_method baseline &> '10_400.txt' &
srun python3 run_all_plot.py -p 15 -m 600 --update_method trimmed_mean --alg_method baseline &> '15_600.txt' &
# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
echo 'done'     
    
