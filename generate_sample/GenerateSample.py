from openslide import OpenSlide
import numpy as np
from skimage import morphology
from scipy import ndimage as ndi
import multiprocessing.dummy as multiprocessing
from multiprocessing import cpu_count
import cv2
import os

block_size = 512
overlab_ratio = 4
level = 0

def get_position_list(ors):
    overlab_size = block_size // overlab_ratio
    width , height = ors.level_dimensions[level]
    print(width , height)
    wmin , wmax = 10000 , 76784
    hmin , hmax = 4608 , 70000
    wnum_block = np.int32((wmax - wmin + 1  - block_size) / (block_size - overlab_size) + 1)
    hnum_block = np.int32((hmax - hmin + 1 - block_size) / (block_size- overlab_size) + 1)
    print('blocks : ' , wnum_block * hnum_block)
    block_position_list = []
    for w in range(0, wnum_block):
        for h in range(0, hnum_block):
            wx = wmin + w * (block_size - overlab_size)
            hy = hmin + h * (block_size - overlab_size)
            block_position_list.append((wx , hy))
    return block_position_list

def get_img_from_position(orses , position , block_size , level , cz , result_parh):
    threColor = 8
    threVol = 62.5 * 4 ** (3 - level)
    imgs = []
    flag = True
    for ors in orses:

        img = ors.read_region(position , level , (block_size , block_size))
        img = np.array(img)
        img = img[: , : , 0 : 3]
        img = img[: , : , ::-1]

        wj1 = img.max(axis=2)
        wj2 = img.min(axis=2)
        wj3 = wj1 - wj2
        img_bin = wj3 > threColor
        img_bin = morphology.remove_small_objects(img_bin, min_size = threVol)
        img_bin = ndi.binary_fill_holes(img_bin > 0)
        imgs.append(img)
        if np.sum(img_bin) < 512 * 512 / 3:
            flag = False

    if flag :
        img = cv2.hconcat((imgs))
        cv2.imwrite(result_parh + cz + '_' + str(position[0]) + '_' + str(position[1]) + '.tif' , img)


def get_imgs(tiff_paths , result_path , cz):
    orses = []
    for tiff_path in tiff_paths:
        ors = OpenSlide(tiff_path)
        orses.append(ors)
    position_list = get_position_list(ors)

    pool = multiprocessing.Pool(int(cpu_count() / 5))

    for position in position_list:
        pool.apply_async(get_img_from_position , args = (orses, position, block_size, level , cz , result_path))

    pool.close()
    pool.join()
    for ors in orses:
        ors.close()

def save_data_by_position(ors_0  ,ors_e , source_path , target_path , cz , position):

    img_0 = np.array(ors_0.read_region(position , 0 , (512 , 512)))
    img_e = np.array(ors_e.read_region(position , 0 , (512 , 512)))

    img_0 , img_e = img_0[: , : , 0 : 3] , img_e[: , : , 0 : 3]
    img_0 , img_e = img_0[: , : , ::-1] , img_e[: , : , ::-1]

    img = cv2.hconcat((img_0 , img_e))

    cv2.imwrite(target_path + 'data\\' + cz + '_' + str(position[0]) + '_' + str(position[1]) + '.tif' , img)


def generate_fusion_sample():
    path = 'D:\\20x_and_40x_data\\data\\'
    source_path = 'D:\\20x_and_40x_new\\20x_tiff\\'
    target_path = 'D:\\20x_and_40x_data\\fusion_data\\'
    czs = os.listdir(path)

    names = ['10140015' , '10140018' , '10140064' , '10140071' , '10140074']

    orses = []
    for name in names:
        ors = OpenSlide(source_path + name + '\\' + name + '_Wholeslide_0.tif')
        orses.append(ors)
        ors = OpenSlide(source_path + name + '\\' + name + '_Wholeslide_Extended.tif')
        orses.append(ors)

    pool = multiprocessing.Pool(cpu_count())
    for cz in czs:
        uints = cz.split('_')
        img_name = uints[0]
        position = (np.int32(uints[1]) , np.int32(uints[2][0 : uints[2].find('.tif')]))
        index = names.index(img_name)
        ors_0 = orses[index * 2]
        ors_e = orses[index * 2 + 1]
        pool.apply_async(save_data_by_position , args = (ors_0 , ors_e , source_path , target_path , img_name , position))

    pool.close()
    pool.join()

    for ors in orses:
        ors.close()

if __name__ == '__main__':
    # path = 'F:\\20x_and_40x_new\\20x_tiff\\'
    # slide_names = ['10140074']
    #     #'10140018' , '10140064' , '10140071' , '10140074']
    # temp_result_path = 'D:\\20x_and_40x_data\\data\\'
    # for slide_name in slide_names:
    #     tiff_paths = []
    #     for layer in range(-5 , 6):
    #         tiff_paths.append(path + slide_name + '\\' + slide_name + '_Wholeslide_' + str(layer) + '.tif')
    #     get_imgs(tiff_paths , temp_result_path , cz = slide_name)

    generate_fusion_sample()