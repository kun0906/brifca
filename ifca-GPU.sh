#!/usr/bin/env bash

: ' multiline comment
  running the shell with source will change the current bash path
  e.g., source stat.sh

  check cuda and cudnn version for tensorflow_gpu==1.13.1
  https://www.tensorflow.org/install/source#linux

  https://researchcomputing.princeton.edu/support/knowledge-base/pytorch
  $ ssh <YourNetID>@della-gpu.princeton.edu  # also adroit or stellar
  $ module load anaconda3/2023.3
  $ CONDA_OVERRIDE_CUDA="11.2" conda create --name ifca_torch "pytorch==2.0*=cuda11*" torchvision -c conda-forge
  $ conda activate ifca_torch
  $ conda install --file requirement.txt
  $ srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=2:00:00 --pty bash -i
  $ python -V     % Python 3.11.3


  # GPU Utilization
  squeue -u $USER
  ssh della-lXXgYY
  watch -n 1 gpustat
'
#ssh ky8517@della-gpu.princeton.edu
cd /scratch/gpfs/ky8517/ifca/mnist
module load anaconda3/2023.3
conda activate ifca_torch

##srun --mem=40G --time=4:00:00 --pty bash -i
#srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=2:00:00 --pty bash -i
