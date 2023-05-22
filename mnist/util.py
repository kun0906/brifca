import torch
import numpy as np
import time
from datetime import datetime

def test():
    pass


def timer(func):
	# This function shows the execution time of
	# the function object passed
	def wrap_func(*args, **kwargs):
		t1 = time.time()
		print(f'- {func.__name__}() starts at {datetime.now()}')
		result = func(*args, **kwargs)
		t2 = time.time()
		print(f'+ {func.__name__}() ends at {datetime.now()}')
		print(f'** Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
		return result

	return wrap_func


def dict_string(my_dict):
    str_list = []
    for key in my_dict:
        if type(my_dict[key]) == float:
            str_list.append(key + f"_{my_dict[key]:.6f}")
        else:
            str_list.append(key + f"_{my_dict[key]}")

    return "_".join(str_list)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

    # use: list(product_dict(**mydict))


def chunk(a, i, n):
    a2 = chunkify(a, n)
    return a2[i]

def chunkify(a, n):
    # splits list into even size list of lists
    # [1,2,3,4] -> [1,2], [3,4]

    k, m = divmod(len(a), n)
    gen = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(gen)


if __name__ == '__main__':
    start_time = time.time()
    test()
    duration = (time.time() - start_time)
    print("---train_cluster Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))