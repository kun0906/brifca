import torch
import numpy as np
import time

def test():

    pass


def random_normal_tensor(size, loc = 0, scale = 1):
    # generate normal distribution with mu = loc and std = scale.
    return torch.randn(size) * scale + loc



if __name__ == '__main__':
    start_time = time.time()
    test()
    duration = (time.time() - start_time)
    print("---train_cluster Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))