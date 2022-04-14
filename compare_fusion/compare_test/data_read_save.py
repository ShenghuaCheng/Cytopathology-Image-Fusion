import cv2
import numpy as np
import multiprocessing.dummy as multi
from multiprocessing import cpu_count
from tqdm import tqdm

test = '/mnt/sda2/20x_and_40x_data/test.txt'

def read_item(path , line , layers):
    temp = []
    for layer in layers:
        img = cv2.imread(path + line + '_' + str(layer) + '.tif')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp.append(img)
    return temp

# read data from file
def read_data(path , layers = [0] , k = -1):

    with open(test) as test_file:

        imgs = []
        names = []

        pool = multi.Pool(cpu_count())
        for line in tqdm(test_file):

            line = line[ : line.find('.tif')]
            names.append(line)
            temp = pool.apply_async(read_item , args = (path , line , layers)).get()

            temp = np.concatenate(temp , axis = 2)
            temp = np.transpose(temp , [2 , 0 , 1])

            imgs.append(temp)

            if k >= 0:
                k -= 1
                if k == 0:
                    break

        pool.close()
        pool.join()

        return imgs , names

# save data
def save_data(path , imgs , names ,norm_way = 1):

    if norm_way == 1:
        imgs = np.uint8(imgs * 255)
    else:
        imgs = np.uint8((imgs + 1) * 127.5)

    imgs = np.transpose(imgs , [0 , 2 , 3 , 1])

    for (name , img) in zip(names , imgs):

        img = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
        cv2.imwrite(path + name + '.tif' , img)

# if __name__ == '__main__':
#     imgs , names = read_data(path , layers = [-2 , -1 , 0 , 1 , 2])
#     print(names)