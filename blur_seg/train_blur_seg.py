import keras.backend as K
from blur_seg.ResNetASPP import ResNetASSP
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import tensorflow as tf
from blur_seg.read_data import DataRead
import random
import numpy as np
# import scipy.ndimage as ndi
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def generate_model(input_shape , weight_path = None , gpu_num = 1):
    K.clear_session()
    classes = 2
    if gpu_num == 1:
            model = ResNetASSP(input_shape=input_shape, weight_decay=0., batch_momentum=0.99, classes=classes)
            if weight_path != None:
                model.load_weights(weight_path)
            model.compile(optimizer=Adam(lr=0.001), loss=["categorical_crossentropy"], metrics=["categorical_accuracy"])
            return model
    else:
        with tf.device('/cpu:0'):
            model = ResNetASSP(input_shape=input_shape, weight_decay=0., batch_momentum=0.99, classes=classes)
            model.compile(optimizer=Adam(lr=0.001), loss=["categorical_crossentropy"], metrics=["categorical_accuracy"])
            if weight_path != None:
                model.load_weights(weight_path)
        parallel_model = multi_gpu_model(model, gpus=gpu_num)
        parallel_model.compile(optimizer=Adam(lr=0.01), loss=["categorical_crossentropy"],metrics=["categorical_accuracy"])
        return parallel_model

def train_model():
    input_shape = (512, 512, 3)

    gpu_num = 1
    batch_size = 6
    train_epoch = 50 #首先训练50轮次
    save_middle_log = 25
    class_weight = [1.0 , 0.5]
    model = generate_model(input_shape, weight_path = None , gpu_num=gpu_num)

    train_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train\\'
    test_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\test\\'

    middle_result = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\middle_result\\'

    dr = DataRead(train_path , test_path)

    train_imgs, train_masks = dr.read_data(train_test_flag = 0)
    test_imgs, test_masks = dr.read_data(train_test_flag = 1)

    train_list = [x for x in range(0 , len(train_imgs))]
    test_list = [x for x in range(0 , len(test_imgs))]

    print(np.shape(train_imgs) , np.shape(test_imgs))

    for epoch in range(1 , train_epoch + 1):
        random.shuffle(train_list)

        train_imgs , train_masks = train_imgs[train_list] , train_masks[train_list]

        for batch_i in range(0 , int(len(train_imgs) / batch_size)):
            temp_imgs = train_imgs[batch_i * batch_size : (batch_i + 1) * batch_size]
            temp_masks = train_masks[batch_i * batch_size : (batch_i + 1) * batch_size]
            loss = model.train_on_batch(temp_imgs , temp_masks , class_weight = class_weight)
            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f , acc.: %.2f%%]" % (epoch , train_epoch ,
                                                                             batch_i , int(len(train_imgs) / batch_size),
                                                                             loss[0], 100 * loss[1]))

            random.shuffle(test_list)
            if batch_i % save_middle_log == 0: #every 25 batch save middle result
                save_middle_result(model , middle_result , test_imgs[test_list[0 : 10]] , test_masks[test_list[0 : 10]] , epoch , batch_i)


def save_middle_result(model , middle_result , imgs , masks , epoch , batch_i):

    pre_masks = model.predict_on_batch(imgs)

    for k in range(0 , len(imgs)):
        pre_mask = pre_masks[k , : , : , 0] > 0.5
        # pre_mask = ndi.binary_fill_holes(pre_mask) #no need to fill holes

        contours , _ = cv2.findContours(np.uint8(pre_mask * 255) , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        img = np.uint8((imgs[k] / 2 + 0.5) * 255)

        cv2.drawContours(img , contours , -1 , (255 , 0 , 0) , 1)

        mask = np.uint8(masks[k , : , : , 0] * 255)
        pre_mask = np.uint8(pre_mask * 255)
        mask = np.stack([mask , mask , mask] , axis = 2)
        pre_mask = np.stack([pre_mask , pre_mask , pre_mask] , axis = 2)

        combined_img = cv2.hconcat((img , pre_mask , mask))

        cv2.imwrite(middle_result + 'imgs_log\\' + str(epoch) + '_' + str(batch_i) + '_' + str(k) + '.tif' , combined_img)

    model.save_weights(middle_result + 'weights_log\\' + str(epoch) + '_' + str(batch_i) + '.h5')

if __name__ == '__main__':
    train_model()