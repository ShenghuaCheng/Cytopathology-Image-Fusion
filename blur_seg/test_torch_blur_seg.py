from blur_seg.read_data import DataRead
import numpy as np
import torch
import cv2
from model.torch_model_resnet_aspp import ResNetASPP

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def torch_tensor_to_numpy(data):
    data = data.cpu().detach().numpy()
    data = np.transpose(data , [0 , 2 , 3 , 1])
    return data

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
        img = img.copy()
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

def test_model(mask_test_imgs , mask_test_masks , degra_test_imgs , degra_test_masks , model_path , model):
    torch.cuda.empty_cache()
    model_blur_seg = ResNetASPP(classes=2, batch_momentum=0.99)
    model_blur_seg.load_state_dict(torch.load(model_path + model))
    model_blur_seg.cuda()
    with torch.no_grad():
        mask_test_imgs = mask_test_imgs.cuda()
        pre_mask_result = model_blur_seg(mask_test_imgs)
        iou_m, pre_m = save_middle_result(result_path,
                                          torch_tensor_to_numpy(mask_test_imgs),
                                          torch_tensor_to_numpy(mask_test_masks),
                                          torch_tensor_to_numpy(pre_mask_result),
                                          model, 'mask')

        degra_test_imgs = degra_test_imgs.cuda()

        test_nums = 50
        pre_degra_result = []
        for i in range(0, int(len(degra_test_imgs) / test_nums)):
            temp_degra_result = model_blur_seg(degra_test_imgs[i * test_nums: (i + 1) * test_nums, :, :, :])
            temp_degra_result = torch_tensor_to_numpy(temp_degra_result)
            pre_degra_result.append(temp_degra_result)

        pre_degra_result = np.concatenate(pre_degra_result, axis=0)
        iou_d, pre_d = save_middle_result(result_path,
                                          torch_tensor_to_numpy(degra_test_imgs),
                                          torch_tensor_to_numpy(degra_test_masks),
                                          pre_degra_result, model, 'degra')

    print(iou_m, pre_m, iou_d, pre_d)
    del model_blur_seg

if __name__ == '__main__':
    model_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\middle_result\\weights_log\\'

    test_models = ['44_225.pth' , '44_300.pth' , '44_450.pth' ,
                   '44_525.pth' , '44_675.pth' , '44_1075.pth' , '44_1125.pth']

    train_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\train\\'
    test_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\test\\'

    result_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\test_result\\'


    dr = DataRead(train_path , test_path)

    test_imgs , test_masks = dr.read_data(train_test_flag = 1 , torch_keras_flag = 1)

    mask_test_imgs , mask_test_masks = test_imgs[0 : 78] , test_masks[0 : 78]
    degra_test_imgs, degra_test_masks = test_imgs[78 : ], test_masks[78 : ]

    for model in test_models:
        test_model(mask_test_imgs, mask_test_masks, degra_test_imgs, degra_test_masks, model_path, model)