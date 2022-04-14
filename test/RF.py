import os
import shutil
from multiprocessing import dummy
from multiprocessing import cpu_count

def item_move(source_path , target_path):

    shutil.copy(source_path ,target_path)


def sample_move():
    test_path = r'X:\GXB\20x_and_40x_data\test_1.txt'

    source_p = r'V:\train_new\compare_new\gen_-1-1_fusion\our_no_BM'
    target_p = r'X:\GXB\20x_and_40x_data\test_result\compares_new\gen_-1-1_fusion\our_no_BM'

    with open(test_path) as test:

        pool = dummy.Pool(int(cpu_count() // 2))
        for line in test:
            line = line[ : line.find('.tif')]
            line += '.tif'

            pool.apply_async(item_move , args = (source_p + '/' + line , target_p + '/' + line))

        pool.close()
        pool.join()


if __name__ == '__main__':
    sample_move()