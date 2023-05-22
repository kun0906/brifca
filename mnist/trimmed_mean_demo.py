# import pydevd_pycharm
# # ping nobel.princeton.edu to get ip address
# pydevd_pycharm.settrace('nobel.princeton.edu', port=34567, stdoutToServer=True, stderrToServer=True)


import numpy as np

import scipy
beta = 0.2

params = np.arange(60)
params = params.reshape((5, 3, 4))
print(params, params.shape)
res = scipy.stats.trim_mean(params, proportiontocut=beta, axis=0)
print(res, res.shape)

import pickle
in_file = 'results.pickle'
with open(in_file, 'rb') as f:
    data = pickle.load(f)
print()