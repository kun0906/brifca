import numpy as np

import scipy
beta = 0.2

params = np.arange(60)
params = params.reshape((5, 3, 4))
print(params, params.shape)
res = scipy.stats.trim_mean(params, proportiontocut=beta, axis=0)
print(res, res.shape)
