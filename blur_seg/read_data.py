import cv2
import os
import torch
import numpy as np
import multiprocessing.dummy as mul
from multiprocessing import cpu_count

class DataRead():

    def __init__(self , train_path , test_path):

        self.train_path = train_path
        self.test_path = test_path
        self.origin_data_path = 'X:\\GXB\\20x_and_40x_data\\split_data\\'


    def get_item_mask(self , data_path , mask_path , cz):
        img = cv2.imread(data_path + cz)
        img = (img / 255. - 0.5) * 2
        mask = cv2.imread(mask_path + cz , 0)
        mask = mask / 255.
        t_mask = np.zeros(np.shape(mask) + (2 ,) , dtype = np.float32)
        t_mask[: , : , 0] = mask
        t_mask[: , : , 1] = 1 - mask
        return [img , t_mask]

    def get_item_degra(self , degra_path , cz):
        temp = cv2.imread(degra_path + cz)
        img = temp[: , 0 : 512 , :]
        img = (img / 255. - 0.5) * 2
        mask = temp[: , 1024 : 1536 , 0] / 255.
        t_mask = np.zeros(np.shape(mask) + (2 ,) , dtype = np.float32)
        t_mask[: , : , 0] = mask
        t_mask[: , : , 1] = 1 - mask
        return [img , t_mask]

    def read_data(self , train_test_flag = 0 , torch_keras_flag = 0):

        if train_test_flag == 0:
            path = self.train_path
        else:
            path = self.test_path

        imgs , masks = [] , []
        pool = mul.Pool(cpu_count())

        czs = os.listdir(path + 'mask\\')
        czs = [x for x in czs if x.find('.tif') != -1]
        for cz in czs:
            t = pool.apply_async(self.get_item_mask , args = (self.origin_data_path , path + 'mask\\' , cz))
            imgs.append(t.get()[0])
            masks.append(t.get()[1])

        czs = os.listdir(path + 'degra\\')
        czs = [x for x in czs if x.find('.tif') != - 1]
        for cz in czs:
            t = pool.apply_async(self.get_item_degra , args = (path + 'degra\\' , cz))
            imgs.append(t.get()[0])
            masks.append(t.get()[1])

        pool.close()
        pool.join()

        imgs , masks = np.array(imgs) , np.array(masks)

        if torch_keras_flag == 0:
            return imgs , masks
        else:
            imgs = np.transpose(imgs , [0 , 3 , 1 , 2]).astype(np.float32)
            masks = np.transpose(masks , [0 , 3 , 1 , 2]).astype(np.float32)
            imgs = torch.from_numpy(imgs)
            masks = torch.from_numpy(masks)
            return imgs , masks

# if __name__ == '__main__':
#     dr = DataRead('X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train\\' , 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\test\\')
#     imgs , masks = dr.read_data(train_test_flag = 1 , torch_keras_flag = 1)
#     print(np.shape(imgs) , np.shape(masks))
#
#     img = imgs[0]
#     mask = masks[0]
#
#     cv2.imwrite('a.tif' , np.uint8((img / 2 + 0.5) * 255))
#     cv2.imwrite('b.tif', np.uint8(mask[: , : , 0] * 255))