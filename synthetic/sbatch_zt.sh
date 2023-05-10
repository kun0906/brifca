#!/bin/bash

#SBATCH --job-name=ifca         # create a short name for your job
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=output/output.txt
#SBATCH --error=output/err.txt
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=zt6264@princeton.edu     # which will cause too much email notification.


module purge
cd /u/zt6264/ifca/synthetic
module load anaconda3/2021.11
conda activate py3104_ifca

pwd
python3 -V
uname -a

PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 run_all.py

#wait


echo 'done'

