import os
import numpy as np
import random
import cv2
import multiprocessing.dummy as multiprocessing
from multiprocessing import cpu_count


def deg():
    txt_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\degra.txt'

    label_path = 'X:\\GXB\\20x_and_40x_data\\label\\'

    result_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\degra_data\\'

    img_size = 512
    with open(txt_path) as name_log:
        for line in name_log:
            line = line[0: line.find('.tif')]
            img = cv2.imread(label_path + line + '.tif')
            mask = np.zeros((512 , 512 , 3) , dtype = np.uint8)
            blur_img = img.copy()
            #随机选择模糊区域的个数0 ~ 10
            blur_region_num = random.randint(0 , 10)

            for blur_region_i in range(0 , blur_region_num):

                #随机选择模糊区域大小
                w_region = random.randint(30 , img_size / 2)
                h_region = random.randint(30 , img_size / 2)

                #随机选择模糊的区域位置
                x = random.randint(0 , img_size - h_region)
                y = random.randint(0 , img_size - w_region)

                #随机选择高斯核的大小，随机选择σ~3σ之间的高斯核半径，先假定σ取1 ，√2 , √3 ,  2 ， √5
                sigma = [1 , np.sqrt(2) , np.sqrt(3) , 2 , np.sqrt(5)]
                sigma_x = sigma[random.randint(0 , 4)]
                sigma_y = sigma[random.randint(0 , 4)]

                h_kernel = int(6 * sigma_x) if int(6 * sigma_x) % 2 == 1 else int(6 * sigma_x) + 1
                w_kernel = int(6 * sigma_y) if int(6 * sigma_y) % 2 == 1 else int(6 * sigma_y) + 1

                print(sigma_x , sigma_y , w_kernel , h_kernel)

                blur_region = blur_img[x : x + h_region , y : y + w_region , :]
                blur_region = cv2.GaussianBlur(blur_region , (h_kernel , w_kernel), sigmaX = sigma_x , sigmaY = sigma_y)
                blur_img[x: x + h_region, y: y + w_region, :] = blur_region
                mask[x: x + h_region, y: y + w_region, :] = 255

            # blur_img = cv2.GaussianBlur(blur_img, (3 , 3) , sigmaX = 0 , sigmaY = 0) #进行平滑操作
            # mask = cv2.GaussianBlur(mask, (3 , 3) , sigmaX = 0 , sigmaY = 0)
            com_img = cv2.hconcat((blur_img , img , mask))

            cv2.imwrite(result_path + line + '.tif' , com_img)


def combine_and_save(img_name , layers , save_combined_path , data_path , label_path):

    img1 = cv2.imread(data_path + img_name + '_' + str(layers[0]) + '.tif')
    img2 = cv2.imread(data_path + img_name + '_' + str(layers[1]) + '.tif')
    img3 = cv2.imread(data_path + img_name + '_' + str(layers[2]) + '.tif')
    img4 = cv2.imread(label_path + img_name + '.tif')

    combined_img1 = cv2.hconcat((img1 , img2))
    combined_img2 = cv2.hconcat((img3 , img4))
    combined_img = cv2.vconcat((combined_img1 , combined_img2))
    cv2.imwrite(save_combined_path + img_name + '.tif' , combined_img)

def generate_combine_img():
    layers = [-2 , 0 , 2]
    save_combined_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\combined_data\\'
    txt_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\mask.txt'

    data_path = 'X:\\GXB\\20x_and_40x_data\\split_data\\'
    label_path = 'X:\\GXB\\20x_and_40x_data\\label\\'

    pool = multiprocessing.Pool(cpu_count())

    with open(txt_path) as name_log:
        for line in name_log:
            line = line[0 : line.find('.tif')]
            pool.apply_async(combine_and_save , args = (line , layers , save_combined_path , data_path , label_path))

    pool.close()
    pool.join()

if __name__ == '__main__':
    deg()
    # generate_combine_img()