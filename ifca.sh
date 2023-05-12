#!/usr/bin/env bash

: ' multiline comment
  running the shell with source will change the current bash path
  e.g., source stat.sh

  check cuda and cudnn version for tensorflow_gpu==1.13.1
  https://www.tensorflow.org/install/source#linux
'
#ssh ky8517@tiger.princeton.edu
cd /scratch/gpfs/ky8517/ifca
module load anaconda3/2021.11

srun --mem=40G --time=4:00:00 --pty bash -i
#srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
#cd /scratch/gpfs/ky8517/ifca
#module load anaconda3/2021.11
conda activate py3104_ifca

