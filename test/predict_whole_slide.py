import os
import random
from multiprocessing import dummy
from multiprocessing import cpu_count
import shutil


def cp_file(source , target):
    shutil.copy(source , target)

path = r'X:\GXB\20x_and_40x_data\data'

source_path = r'X:\GXB\20x_and_40x_data\split_data'
target_path = r'V:\data\split_data'

names = os.listdir(path)
names = [x for x in names if x.find('.tif') != -1]

pool = dummy.Pool(int(cpu_count() // 2))

for name in names:

    name = name[ : name.find('.tif')]

    name1 = name + '_0.tif'
    name2 = name + '_-1.tif'
    name3 = name + '_1.tif'

    # pool.apply_async(cp_file , args = (source_path + '/' + name1 , target_path + '/' + name1))
    # pool.apply_async(cp_file , args = (source_path + '/' + name2 , target_path + '/' + name2))
    pool.apply_async(cp_file , args = (source_path + '/' + name3 , target_path + '/' + name3))

pool.close()
pool.join()