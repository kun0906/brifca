"""
    https://github.com/PrincetonUniversity/hpc_beginning_workshop/tree/main/cxx/hybrid_multithreaded_parallel

    We want to run a script on multi-nodes, and each node has multi-tasks

    https://stackoverflow.com/questions/58648721/how-to-get-slurm-task-id-in-program

"""
import datetime
print(datetime.datetime.now())
import numpy as np
import subprocess
import os

print('SLURM_JOB_NAME: ', os.environ['SLURM_JOB_NAME'], 'task2')
print('SLURM_JOB_ID: ', os.environ['SLURM_JOB_ID'], 'task2')
print('SLURM_PROCID: ', os.environ['SLURM_PROCID'], 'task2')
print('SLURM_PROCID by getenv:', os.getenv('SLURM_PROCID'), 'task2')

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
for i in range(1000000):
    X_i = np.eye(d) * i
    res = X_i @ res

print('finish')
print(datetime.datetime.now())
