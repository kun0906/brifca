"""
    https://github.com/PrincetonUniversity/hpc_beginning_workshop/tree/main/cxx/hybrid_multithreaded_parallel

    We want to run a script on multi-nodes, and each node has multi-tasks

    https://stackoverflow.com/questions/58648721/how-to-get-slurm-task-id-in-program

"""
import datetime
print(datetime.datetime.now())

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
#                     action='store_true', help='force')
parser.add_argument("--array_task_id", type=int, default=50)
args = parser.parse_args()

print(args)

import numpy as np
import subprocess
import os

print('SLURM_JOB_NAME: ', os.environ['SLURM_JOB_NAME'])
print('SLURM_JOB_ID: ', os.environ['SLURM_JOB_ID'])
print('SLURM_PROCID: ', os.environ['SLURM_PROCID'])
print('SLURM_PROCID by getenv:', os.getenv('SLURM_PROCID'))

cmd = f'uname -a'
ret = subprocess.run(cmd, shell=True)
print(cmd, ret)

import platform
print('platform: ', platform.node())

import socket
print('socket: ', socket.gethostname())

import multiprocessing
cpu_cores=multiprocessing.cpu_count()
print(f'cpu_cores: {cpu_cores}')

d = 100
res = np.eye(d)
for i in range(100000):
    X_i = np.eye(d) * i
    res = X_i @ res

print('finish')
print(datetime.datetime.now())