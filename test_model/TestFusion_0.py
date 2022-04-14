import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
import cv2
from model.torch_model_resnet_aspp import ResNetASPP
from train_and_test.public_code import *
from model.model import *
import matplotlib.pyplot as plt

#对于模糊识别模型，需要预先加载进来
mask_threshold = 0.5
model_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\middle_result\\weights_log\\44_1125.pth'
model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
model_blur_seg.load_state_dict(torch.load(model_path))
model_blur_seg.cuda()


color = ['red', 'blue', 'green', 'black', 'cyan' , 'magenta' , 'yellow']
log_tag = ['unet_patch64', 'unet_patch64_catblur', 'unet_patch64_maeblur', 'unet_patch32' , 'unet_blur' , '0_layer', '11_fusion']

def test_models():
    path_unet_patch64 = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_unet_patch64\\'
    path_unet_patch64_catblur = 'V:\\fusion_0_catblur_unet_patch64\\checkpoints\\'
    path_unet_patch64_maeblur = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_layer_blur_mae\\'

    models_unet_patch64 = ['netG_epoch_8_5651.pth' , 'netG_epoch_8_5681.pth' , 'netG_epoch_8_5703.pth' , 'netG_epoch_8_5921.pth' ,
                            'netG_epoch_8_5945.pth']
    models_unet_patch64_catblur = ['netG_epoch_8_5895.pth' , 'netG_epoch_8_5871.pth' , 'netG_epoch_8_5921.pth' , 'netG_epoch_8_5757.pth' ,
                            'netG_epoch_8_5759.pth']
    models_unet_patch_64_maeblur = ['netG_epoch_8_5857.pth' , 'netG_epoch_8_5791.pth' , 'netG_epoch_8_5486.pth' , 'netG_epoch_8_5344.pth' ,
                            'netG_epoch_8_5264.pth']

    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_0_test_blur\\models\\'


    layers = [0]
    img_num = 100

    datas , labels = read_imgs(layers , img_num = img_num)
    print(np.shape(datas) , np.shape(labels))

    ssim_avgt, cc_avgt = [], []

    for index in range(0, len(models_unet_patch64)):

        model_unet_patch64 = Light(3, 3)
        model_unet_patch64.load_state_dict(torch.load(path_unet_patch64 + models_unet_patch64[index]))
        model_unet_patch64.cuda()

        model_unet_patch64_catblur = Light(4, 3)
        model_unet_patch64_catblur.load_state_dict(torch.load(path_unet_patch64_catblur + models_unet_patch64_catblur[index]))
        model_unet_patch64_catblur.cuda()

        model_unet_patch64_maeblur = Light(3, 3)
        model_unet_patch64_maeblur.load_state_dict(torch.load(path_unet_patch64_maeblur + models_unet_patch_64_maeblur[index]))
        model_unet_patch64_maeblur.cuda()

        ssim_avg = np.array([[0., 0] , [0., 0] , [0., 0]])
        cc_avg = np.array([[0., 0] , [0., 0] , [0., 0]])
        k = 0
        for data, label in zip(datas, labels):
            data = [data]
            label = [label]
            data = torch.from_numpy(np.array(data))
            label = torch.from_numpy(np.array(label))
            print(np.shape(data))
            data = data.cuda()
            with torch.no_grad():
                gen_unet_patch64 = model_unet_patch64(data)
                gen_unet_patch64_maeblur = model_unet_patch64_maeblur(data)

                blur_mask = model_blur_seg(data)
                blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
                blur_mask = blur_mask.reshape((blur_mask.size(0), 1, blur_mask.size(1), blur_mask.size(2)))
                data_blur = torch.cat([data, blur_mask], dim=1)
                gen_unet_patch64_catblur = model_unet_patch64_catblur(data_blur)

                ssim_avg1, cc_avg1 = save_result_and_evaluate_2(layers, result_path + 'unet_patch64\\' + models_unet_patch64[index] + '_',
                                                                log_tag[0],
                                                                data[0], gen_unet_patch64[0],
                                                                label[0], k)
                ssim_avg2, cc_avg2 = save_result_and_evaluate_2(layers, result_path + 'unet_patch64_catblur\\' + models_unet_patch64_catblur[index] + '_',
                                                                log_tag[1],
                                                                data[0],
                                                                gen_unet_patch64_catblur[0],
                                                                label[0], k)
                ssim_avg3, cc_avg3 = save_result_and_evaluate_2(layers, result_path + 'unet_patch64_maeblur\\' + models_unet_patch_64_maeblur[index] + '_',
                                                                log_tag[2],
                                                                data[0],
                                                                gen_unet_patch64_maeblur[0],
                                                                label[0], k)

                ssim_avg += np.array([ssim_avg1 , ssim_avg2 , ssim_avg3])
                cc_avg += np.array([cc_avg1 , cc_avg2 , cc_avg3])
            k += 1

        ssim_avg /= k
        cc_avg /= k

        ssim_avgt.append(ssim_avg)
        cc_avgt.append(cc_avg)

        print(ssim_avg , cc_avg)

    np.save('ssim1', np.array(ssim_avgt))
    np.save('cc1', np.array(cc_avgt))


def test_imgs():
    layers = [0]
    img_num = 10

    datas , labels = read_imgs(layers , img_num = img_num)
    print(np.shape(datas) , np.shape(labels))

    model_path_unet_patch64 = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_unet_patch64\\netG_epoch_8_5651.pth'
    model_path_unet_patch64_catblur = 'V:\\fusion_0_catblur_unet_patch64\\checkpoints\\netG_epoch_8_5651.pth'
    model_path_unet_patch64_maeblur = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_layer_blur_mae\\netG_epoch_8_5857.pth'

    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_0_test_blur\\imgs\\'

    model_unet_patch64 = Light(3 , 3)
    model_unet_patch64.load_state_dict(torch.load(model_path_unet_patch64))
    model_unet_patch64.cuda()

    model_unet_patch64_catblur = Light(4 , 3)
    model_unet_patch64_catblur.load_state_dict(torch.load(model_path_unet_patch64_catblur))
    model_unet_patch64_catblur.cuda()

    model_unet_patch64_maeblur = Light(3, 3)
    model_unet_patch64_maeblur.load_state_dict(torch.load(model_path_unet_patch64_maeblur))
    model_unet_patch64_maeblur.cuda()

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
            gen_unet_patch64 = model_unet_patch64(data)
            gen_unet_patch64_maeblur = model_unet_patch64_maeblur(data)

            blur_mask = model_blur_seg(data)
            blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
            blur_mask = blur_mask.reshape((blur_mask.size(0), 1, blur_mask.size(1), blur_mask.size(2)))
            data_blur = torch.cat([data, blur_mask], dim=1)
            gen_unet_patch64_catblur = model_unet_patch64_catblur(data_blur)

            ssim_avg1, cc_avg1 = save_result_and_evaluate_2(layers, result_path,
                                                                                   log_tag[0],
                                                                                   data[0], gen_unet_patch64[0],
                                                                                   label[0], k)
            ssim_avg2, cc_avg2 = save_result_and_evaluate_2(layers, result_path,
                                                                               log_tag[1],
                                                                               data[0],
                                                                               gen_unet_patch64_catblur[0],
                                                                               label[0], k)
            ssim_avg3, cc_avg3= save_result_and_evaluate_2(layers, result_path,
                                                                               log_tag[2],
                                                                               data[0],
                                                                               gen_unet_patch64_maeblur[0],
                                                                               label[0], k)


            ssim_avg.append([ssim_avg1 , ssim_avg2 , ssim_avg3])
            cc_avg.append([cc_avg1 , cc_avg2 , cc_avg3])
        k += 1
    ssim_avg = np.array(ssim_avg)
    cc_avg = np.array(cc_avg)

    np.save('ssim1', np.array(ssim_avg))
    np.save('cc1', np.array(cc_avg))


def draw_map():
    ssim = np.load('ssim1.npy')
    cc = np.load('cc1.npy')

    # print(np.shape(en) , np.shape(sd) , np.shape(ssim) , np.shape(cc) , np.shape(sf) , np.shape(vif))

    x = [t for t in range(0 , 5)]

    plt.plot(x , cc[ : , 0 , 1], c=color[0], label = log_tag[0])
    plt.plot(x, cc[:, 1 , 1], c=color[1], label = log_tag[1])
    plt.plot(x , cc[: , 2 , 1] , c = color[2] , label = log_tag[2])
    plt.plot(x, cc[:, 0, 0], c=color[5], label=log_tag[5])
    if False:
        plt.plot(x, ssim[:, 4 , 2], c=color[6], label=log_tag[6])
        # plt.plot(x, ssim[:, 0], c='magenta', label=tag)

    plt.legend(loc = 'best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("multi_imgs_cc")

    plt.show()


if __name__ == '__main__':
    # test_models()
    # test_imgs()
    draw_map()