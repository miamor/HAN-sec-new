from multiprocessing import Pool
import time
def func(arg):
    # time.sleep(0.001)
    return arg

proc_pool = Pool(4)
a = proc_pool.map(func, range(30))
print(a)




from multiprocessing import Pool, TimeoutError
from multiprocessing.pool import ThreadPool
import os

__THREADS_NUM__ = 4

X = []
def func(x, y):
    return x*y

pool = Pool(processes=__THREADS_NUM__)

results = pool.map(func, [1,3,5], 2)
print(results)
