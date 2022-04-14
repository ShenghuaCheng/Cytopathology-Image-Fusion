import numpy as np
import cv2
import os


def fuse_img(path1 , path2 , cz):

    img1 = cv2.imread(path1 + cz)
    img1 = np.array(img1)
    img1 = img1[192 : 320 , 192 : 320 , :]

    cv2.imwrite(path2 + cz , img1)

if __name__ == '__main__':

    path1 = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\label\\'
    path2 = 'E:\\kernel_smooth\\test_data_b\\'

    czs = os.listdir(path1)
    czs = [x for x in czs if x.find('.tif') != -1]

    for cz in czs:
        fuse_img(path1 , path2 , cz)