import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import cv2
import copy
from model.torch_model_resnet_aspp import ResNetASPP
from train_and_test.public_code import *
from model.model import *
import matplotlib.pyplot as plt
from compare_fusion.compare_models import *

color = ['red', 'blue', 'green', 'black', 'cyan' , 'magenta' , 'yellow']
log_tag = ['pixel2pixel', 'fusion_gan', 'unet_patch64_catblur', 'unet_patch32' , 'unet_blur' , '0_layer', '11_fusion']


#对于模糊识别模型，需要预先加载进来
mask_threshold = 0.5
model_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\middle_result\\weights_log\\44_1125.pth'
model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
model_blur_seg.load_state_dict(torch.load(model_path))
model_blur_seg.cuda()

def test_models():
    path_piexl2pixel = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\pixel2pixel_0_layer\\'
    path_fusionGan = 'D:\\20x_and_40x_data\\checkpoints\\fusionGan_0_layer\\'
    path_unet_patch64_catblur = 'V:\\fusion_0_catblur_unet_patch64\\checkpoints\\'

    models_piexl2pixel = ['netG_epoch_1_312.pth', 'netG_epoch_1_589.pth', 'netG_epoch_2_775.pth', 'netG_epoch_2_797.pth',
                 'netG_epoch_2_942.pth']
    models_fusionGan = ['netG_epoch_2_1162.pth', 'netG_epoch_2_1163.pth', 'netG_epoch_2_1164.pth', 'netG_epoch_2_1165.pth',
                 'netG_epoch_2_1166.pth']
    models_unet_patch64_catblur = ['netG_epoch_8_5895.pth' , 'netG_epoch_8_5871.pth' , 'netG_epoch_8_5921.pth' , 'netG_epoch_8_5757.pth' ,
                            'netG_epoch_8_5759.pth']

    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compare_models\\models\\'


    layers = [0]
    img_num = 30

    datas , labels = read_imgs(layers , img_num = img_num)
    print(np.shape(datas) , np.shape(labels))

    ssim_avgt, cc_avgt = [], []

    for index in range(0, len(models_piexl2pixel)):

        model_pixel2pixel = Pixel2PixelGenerator(3 , 3)
        model_pixel2pixel.load_state_dict(torch.load(path_piexl2pixel + models_piexl2pixel[index]))
        model_pixel2pixel.cuda()

        model_fusionGan = FusionGanGenerator(3 , 3)
        model_fusionGan.load_state_dict(torch.load(path_fusionGan + models_fusionGan[index]))
        model_fusionGan.cuda()

        model_unet_patch64_catblur = Light(4, 3)
        model_unet_patch64_catblur.load_state_dict(torch.load(path_unet_patch64_catblur + models_unet_patch64_catblur[index]))
        model_unet_patch64_catblur.cuda()

        ssim_avg = np.array([[0., 0] , [0., 0] , [0. , 0.]])
        cc_avg = np.array([[0., 0] , [0., 0] , [0. , 0.]])
        k = 0
        for data, label in zip(datas, labels):
            data = [data]
            label = [label]
            data = torch.from_numpy(np.array(data))
            label = torch.from_numpy(np.array(label))
            print(np.shape(data))
            data = data.cuda()
            with torch.no_grad():
                gen_pixel2pixel = model_pixel2pixel(data)
                gen_fusionGan = model_fusionGan(data)

                temp_data = copy.deepcopy(data)
                temp_data = ((temp_data + 1) * 127.5) / 255. #做适应数据的操作
                blur_mask = model_blur_seg(temp_data)
                blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
                blur_mask = blur_mask.reshape((blur_mask.size(0), 1, blur_mask.size(1), blur_mask.size(2)))
                data_blur = torch.cat([temp_data, blur_mask], dim=1)
                gen_unet_patch64_catblur = model_unet_patch64_catblur(data_blur)
                #将gen_unet_patch64_catblur 归一化到-1~1之间
                gen_unet_patch64_catblur = gen_unet_patch64_catblur * 255. / 127.5 - 1

                ssim_avg1, cc_avg1 = save_result_and_evaluate_2(layers, result_path, log_tag[0] + '\\' + models_piexl2pixel[index],
                                                                                                   data[0], gen_pixel2pixel[0],
                                                                                                   label[0], k)
                ssim_avg2, cc_avg2 = save_result_and_evaluate_2(layers, result_path, log_tag[1] + '\\' +
                                                                                                   models_fusionGan[index],
                                                                                                   data[0],
                                                                                                   gen_fusionGan[0],
                                                                                                   label[0], k)

                ssim_avg3, cc_avg3 = save_result_and_evaluate_2(layers, result_path, log_tag[2] + '\\' +
                                                                                                   models_unet_patch64_catblur[index],
                                                                                                   data[0],
                                                                                                   gen_unet_patch64_catblur[0],
                                                                                                   label[0], k )

                ssim_avg += np.array([ssim_avg1 , ssim_avg2 , ssim_avg3])
                cc_avg += np.array([cc_avg1 , cc_avg2 , cc_avg3])
            k += 1
        ssim_avg /= k
        cc_avg /= k

        ssim_avgt.append(ssim_avg)
        cc_avgt.append(cc_avg)

        print(ssim_avg , cc_avg)

    np.save('ssim', np.array(ssim_avgt))
    np.save('cc', np.array(cc_avgt))


def test_imgs():
    layers = [0]
    img_num = 10

    datas , labels = read_imgs(layers , img_num = img_num)
    print(np.shape(datas) , np.shape(labels))


    model_path_piexl2pixel = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\pixel2pixel_0_layer\\netG_epoch_2_797.pth'
    model_path_fusionGan = 'D:\\20x_and_40x_data\\checkpoints\\fusionGan_0_layer\\netG_epoch_2_1165.pth'
    model_path_unet_patch64_catblur = 'V:\\fusion_0_catblur_unet_patch64\\checkpoints\\netG_epoch_8_5757.pth'

    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compare_models\\imgs\\'

    model_pixel2pixel = Pixel2PixelGenerator(3, 3)
    model_pixel2pixel.load_state_dict(torch.load(model_path_piexl2pixel))
    model_pixel2pixel.cuda()

    model_fusionGan = FusionGanGenerator(3, 3)
    model_fusionGan.load_state_dict(torch.load(model_path_fusionGan))
    model_fusionGan.cuda()

    model_unet_patch64_catblur = Light(4, 3)
    model_unet_patch64_catblur.load_state_dict(torch.load(model_path_unet_patch64_catblur))
    model_unet_patch64_catblur.cuda()

    k = 0

    ssim_avg , cc_avg = [] , []
    for data , label in zip(datas , labels):
        data = [data]
        label = [label]
        data = torch.from_numpy(np.array(data))
        label = torch.from_numpy(np.array(label))
        print(np.shape(data))
        data = data.cuda()
        with torch.no_grad():
            gen_pixel2pixel = model_pixel2pixel(data)
            gen_fusionGan = model_fusionGan(data)

            temp_data = copy.deepcopy(data)
            temp_data = ((temp_data + 1) * 127.5) / 255.  # 做适应数据的操作
            blur_mask = model_blur_seg(temp_data)
            blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
            blur_mask = blur_mask.reshape((blur_mask.size(0), 1, blur_mask.size(1), blur_mask.size(2)))
            data_blur = torch.cat([temp_data, blur_mask], dim=1)
            gen_unet_patch64_catblur = model_unet_patch64_catblur(data_blur)
            # 将gen_unet_patch64_catblur 归一化到-1~1之间
            gen_unet_patch64_catblur = gen_unet_patch64_catblur * 255. / 127.5 - 1

            ssim_avg1, cc_avg1 = save_result_and_evaluate_2(layers, result_path,
                                                            log_tag[0],
                                                            data[0], gen_pixel2pixel[0],
                                                            label[0], k)
            ssim_avg2, cc_avg2 = save_result_and_evaluate_2(layers, result_path, log_tag[1],
                                                            data[0],
                                                            gen_fusionGan[0],
                                                            label[0], k)

            ssim_avg3, cc_avg3 = save_result_and_evaluate_2(layers, result_path, log_tag[2] ,
                                                            data[0],
                                                            gen_unet_patch64_catblur[0],
                                                            label[0], k)



            ssim_avg.append([ssim_avg1 , ssim_avg2 , ssim_avg3])
            cc_avg.append([cc_avg1 , cc_avg2 , cc_avg3 ])
        k += 1
    ssim_avg = np.array(ssim_avg)
    cc_avg = np.array(cc_avg)

    np.save('ssim1', np.array(ssim_avg))
    np.save('cc1', np.array(cc_avg))


def draw_map():
    # en = np.load('en1.npy')
    # sd = np.load('sd1.npy')
    ssim = np.load('ssim1.npy')
    cc = np.load('cc1.npy')
    # sf = np.load('sf1.npy')
    # vif = np.load('vif1.npy')

    print(np.shape(ssim) , np.shape(cc))

    x = [t for t in range(0 , 10)]

    plt.plot(x , ssim[ : , 0 , 1], c=color[0], label = log_tag[0])
    plt.plot(x, ssim[:, 1 , 1], c=color[1], label = log_tag[1])
    plt.plot(x , ssim[: , 2 , 1] , c = color[2] , label = log_tag[2])
    # plt.plot(x, ssim[:, 3 , 1], c=color[3], label = log_tag[3])
    # plt.plot(x , ssim[: , 4 , 1] , c = color[4] , label = log_tag[4])
    plt.plot(x, ssim[:, 0, 0], c=color[5], label=log_tag[5])
    if False:
        plt.plot(x, ssim[:, 4 , 2], c=color[6], label=log_tag[6])
        # plt.plot(x, ssim[:, 0], c='magenta', label=tag)

    plt.legend(loc = 'best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("multi_imgs_ssim")

    plt.show()


if __name__ == '__main__':
    # test_models()
    # test_imgs()
    draw_map()