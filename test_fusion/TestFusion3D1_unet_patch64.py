import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
import cv2
from model.model import *
import random
import multiprocessing.dummy as multi
from multiprocessing import cpu_count
import torch

def read_item_by_name(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img = np.float32(img / 255)
    img = np.transpose(img , [2 , 0 , 1])
    return img

def read_imgs(imgs_path):
    imgs_name = os.listdir(imgs_path)
    imgs_name = [x for x in imgs_name if x.find('.tif') != -1]

    read_num = 1000
    random.shuffle(imgs_name)
    imgs_name = imgs_name[ : read_num]

    imgs = []
    pool = multi.Pool(cpu_count())
    for name in imgs_name:
        imgs.append(pool.apply_async(read_item_by_name , args = (imgs_path + name , )).get())
    pool.close()
    pool.join()
    imgs = np.array(imgs)
    return imgs

def test_imgs():

    test_data_path = 'X:\\GXB\\20x_and_40x_data\\other_data_test\\test_our_data\\'
    datas = read_imgs(test_data_path)
    print(np.shape(datas))

    model_path_unet_patch64 = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_unet_patch64\\netG_epoch_8_5703.pth'

    result_path = 'X:\\GXB\\20x_and_40x_data\\other_data_test\\test_our_result\\'

    model_unet_patch64 = Light(3 , 3)
    model_unet_patch64.load_state_dict(torch.load(model_path_unet_patch64))
    model_unet_patch64.cuda()

    k = 0

    for data in datas:
        data = [data]
        data = torch.from_numpy(np.array(data))
        data = data.cuda()
        with torch.no_grad():
            gen = model_unet_patch64(data)

        print(np.shape(data) , np.shape(gen))
        data = data.cpu().detach().numpy()
        gen = gen.cpu().detach().numpy()

        data = cv2.cvtColor(np.transpose(np.uint8(data[0] * 255) , [1 , 2 , 0]) , cv2.COLOR_RGB2BGR)
        gen = cv2.cvtColor(np.transpose(np.uint8(gen[0] * 255), [1, 2, 0]) , cv2.COLOR_RGB2BGR)
        img = cv2.hconcat((data , gen))
        cv2.imwrite(result_path + str(k) + '.tif' , img)

        k += 1


if __name__ == '__main__':
    test_imgs()