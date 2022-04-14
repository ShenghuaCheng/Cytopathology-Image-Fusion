import os
import openslide
import numpy as np
import cv2
import multiprocessing.dummy as multi
from multiprocessing import cpu_count
from skimage import morphology
from scipy import ndimage as ndi


result_path = 'X:\\GXB\\20x_and_40x_data\\shy1_positive\\test_our_data\\'

def save_img(ors , cz , position , block_size , level = 0):
    img = ors.read_region(position , level , (block_size , block_size))
    img = np.array(img)
    img = img[: , : , 0 : 3]
    img = img[: , : , ::-1]

    thre_color = 8
    img_bin = (np.max(img , axis = 2) - np.min(img , axis = 2)) > thre_color

    thre_vol = 62.5 * 4 ** 3
    img_bin = ndi.binary_fill_holes(img_bin > 0)
    img_bin = morphology.remove_small_objects(img_bin, min_size = thre_vol)

    if np.sum(img_bin) > block_size * block_size / 3:
        cv2.imwrite(result_path + cz + '_' + str(position[0]) + '_' + str(position[1]) + '_' + str(level) + '.tif' , img)

def generate_sample(path , cz):

    ors = openslide.OpenSlide(path + cz)

    bin_level = 5
    img = ors.read_region((0 , 0) , bin_level , ors.level_dimensions[bin_level])
    img = np.array(img)
    img = img[: , : , 0 : 3]
    img = img[: , : , ::-1]

    thre_color = 8
    img_bin = (np.max(img , axis = 2) - np.min(img , axis = 2)) > thre_color

    h_min = img_bin.nonzero()[0][0] * 2 ** bin_level + 1000
    h_max = img_bin.nonzero()[0][-1] * 2 ** bin_level - 1000
    w_min = 1000
    w_max = ors.level_dimensions[0][0] - 1000

    block_size = 512
    overlab = 128

    wnum_block = np.int32(((w_max - w_min + 1)  - block_size) / (block_size - overlab) + 1)
    hnum_block = np.int32(((h_max - h_min + 1) - block_size) / (block_size- overlab) + 1)

    pool = multi.Pool(cpu_count())

    for w in range(0, wnum_block):
        for h in range(0, hnum_block):
            wx = w_min + w * (block_size - overlab)
            hy = h_min + h * (block_size - overlab)
            pool.apply_async(save_img , args = (ors , cz , (wx , hy) , block_size))

    pool.close()
    pool.join()

    ors.close()

if __name__ == '__main__':
    mrxses_path = 'X:\\TCT\\TCTDATA\\Shengfuyou_1th\\Positive\\'
    czs = os.listdir(mrxses_path)
    czs = [x for x in czs if x.find('.mrxs') != -1]

    for i in range(0 , 10):
        generate_sample(mrxses_path , czs[i])