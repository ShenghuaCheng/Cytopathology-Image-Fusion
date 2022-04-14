from blur_seg.train_blur_seg import generate_model
from blur_seg.read_data import DataRead
import numpy as np
import keras.backend as k
import cv2
import os


def save_middle_result(middle_result_path , imgs , masks , pre_masks , weight , c):

    iou_mean , pre_mean = 0 , 0
    for k in range(0 , len(imgs)):
        pre_mask = pre_masks[k , : , : , 0] > 0.5
        lab_mask = masks[k , : , : , 0] > 0

        if np.sum(pre_mask) == 0 and np.sum(lab_mask) == 0:
            iou  , precision = 0 , 0
        else:
            iou = np.sum(pre_mask * lab_mask) / np.sum(pre_mask + lab_mask)
            if np.sum(lab_mask) == 0:
                precision = 1 - (np.sum(pre_mask) / (512 * 512))
            else:
                precision = np.sum(pre_mask * lab_mask) / np.sum(lab_mask)

        iou_mean += iou
        pre_mean += precision

        contours , _ = cv2.findContours(np.uint8(pre_mask * 255) , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        img = np.uint8((imgs[k] / 2 + 0.5) * 255)

        cv2.drawContours(img , contours , -1 , (255 , 0 , 0) , 1)

        mask = np.uint8(masks[k , : , : , 0] * 255)
        pre_mask = np.uint8(pre_mask * 255)
        mask = np.stack([mask , mask , mask] , axis = 2)
        pre_mask = np.stack([pre_mask , pre_mask , pre_mask] , axis = 2)

        combined_img = cv2.hconcat((img , pre_mask , mask))

        if not os.path.exists(middle_result_path + weight + '\\' + c + '\\'):
            os.makedirs(middle_result_path + weight + '\\' + c + '\\')

        cv2.imwrite(middle_result_path + weight + '\\' + c + '\\' +  str(k) + '.tif' , combined_img)

    return iou_mean / len(imgs) , pre_mean / len(imgs)

if __name__ == '__main__':
    model_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\middle_result\\weights_log\\'

    test_models = ['50_425.h5' , '50_575.h5' , '50_650.h5' , '50_700.h5' , '50_800.h5']

    train_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train\\'
    test_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\test\\'

    result_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\test_result\\'


    dr = DataRead(train_path , test_path)

    test_imgs , test_masks = dr.read_data(train_test_flag = 1)

    mask_test_imgs , mask_test_masks = test_imgs[0 : 81] , test_masks[0 : 81]
    degra_test_imgs, degra_test_masks = test_imgs[81 : ], test_masks[81 : ]

    for model in test_models:
        k.clear_session()
        model_blur = generate_model((512 , 512 , 3) , weight_path = (model_path + model) , gpu_num = 1)
        pre_mask_result = model_blur.predict(mask_test_imgs)
        iou_m , pre_m = save_middle_result(result_path, mask_test_imgs, mask_test_masks, pre_mask_result, model, 'mask')

        pre_degra_result = model_blur.predict(degra_test_imgs)
        iou_d, pre_d = save_middle_result(result_path, degra_test_imgs, degra_test_masks, pre_degra_result, model, 'degra')

        print(iou_m , pre_m , iou_d , pre_d)