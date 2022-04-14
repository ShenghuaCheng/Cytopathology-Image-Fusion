import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
log_tag = ['unet', '15srgan_patch16', 'unet_patch64', 'unet_patch32' , 'unet_blur' , '0_layer', '11_fusion']

def test_models():
    path_unet = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_layer\\'
    path_15srgan_patch = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_layer_patch\\'
    path_unet_patch_64 = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_unet_patch64\\'
    path_unet_patch_32 = 'V:\\fusion_0_unet_patch32\\checkpoints_unet\\'
    path_unet_blur = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_layer_c\\'

    models_unet = ['netG_epoch_8_5883.pth', 'netG_epoch_8_5873.pth', 'netG_epoch_8_5854.pth', 'netG_epoch_8_5839.pth',
                 'netG_epoch_8_5807.pth']
    models_15srgan_patch = ['netG_epoch_7_4858.pth', 'netG_epoch_7_4829.pth', 'netG_epoch_7_4816.pth', 'netG_epoch_7_4815.pth',
                 'netG_epoch_7_4893.pth']
    models_unet_patch_64 = ['netG_epoch_8_5651.pth' , 'netG_epoch_8_5681.pth' , 'netG_epoch_8_5703.pth' , 'netG_epoch_8_5921.pth' ,
                            'netG_epoch_8_5945.pth']
    models_unet_patch_32 = ['netG_epoch_8_5956.pth' , 'netG_epoch_8_5953.pth' , 'netG_epoch_8_5951.pth' , 'netG_epoch_8_5949.pth' ,
                            'netG_epoch_8_5942.pth']
    models_unet_blur = ['netG_epoch_8_5958.pth' , 'netG_epoch_8_5951.pth' , 'netG_epoch_8_5942.pth' , 'netG_epoch_8_5939.pth' ,
                            'netG_epoch_8_5927.pth']

    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_0_layer_test_all\\models\\'


    layers = [0]
    img_num = 30

    datas , labels = read_imgs(layers , img_num = img_num)
    print(np.shape(datas) , np.shape(labels))

    en_avgt, sd_avgt, ssim_avgt, cc_avgt, sf_avgt, vif_avgt = [], [], [], [], [], []

    for index in range(0, len(models_unet)):

        model_unet = Light(3, 3)
        model_unet.load_state_dict(torch.load(path_unet + models_unet[index]))
        model_unet.cuda()

        model_15srgan_patch = SingleLayerGenerator(15)
        model_15srgan_patch.load_state_dict(torch.load(path_15srgan_patch + models_15srgan_patch[index]))
        model_15srgan_patch.cuda()

        model_unet_patch_64 = Light(3, 3)
        model_unet_patch_64.load_state_dict(torch.load(path_unet_patch_64 + models_unet_patch_64[index]))
        model_unet_patch_64.cuda()

        model_unet_patch_32 = Light(3, 3)
        model_unet_patch_32.load_state_dict(torch.load(path_unet_patch_32 + models_unet_patch_32[index]))
        model_unet_patch_32.cuda()

        model_unet_blur = Light(4, 3)
        model_unet_blur.load_state_dict(torch.load(path_unet_blur + models_unet_blur[index]))
        model_unet_blur.cuda()

        en_avg = np.array([[0., 0, 0] , [0., 0, 0] , [0., 0, 0] , [0 , 0 , 0] , [0 , 0 , 0]])
        sd_avg = np.array([[0., 0, 0] , [0., 0, 0] , [0., 0, 0] , [0 , 0 , 0] , [0 , 0 , 0]])
        ssim_avg = np.array([[0., 0] , [0., 0] , [0., 0] , [0 , 0] , [0 , 0]])
        cc_avg = np.array([[0., 0] , [0., 0] , [0., 0] , [0 , 0] , [0 , 0]])
        sf_avg = np.array([[0., 0, 0] , [0., 0, 0] , [0., 0, 0] , [0 , 0 , 0] , [0 , 0 , 0]])
        vif_avg = np.array([[0., 0] , [0., 0] , [0., 0] , [0 , 0] , [0 , 0]])
        k = 0
        for data, label in zip(datas, labels):
            data = [data]
            label = [label]
            data = torch.from_numpy(np.array(data))
            label = torch.from_numpy(np.array(label))
            print(np.shape(data))
            data = data.cuda()
            with torch.no_grad():
                gen_unet = model_unet(data)
                gen_15srgan_patch = model_15srgan_patch(data)
                gen_unet_patch_64 = model_unet_patch_64(data)
                gen_unet_patch_32 = model_unet_patch_32(data)

                blur_mask = model_blur_seg(data)
                blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
                blur_mask = blur_mask.reshape((blur_mask.size(0), 1, blur_mask.size(1), blur_mask.size(2)))
                data_blur = torch.cat([data, blur_mask], dim=1)
                gen_unet_blur = model_unet_blur(data_blur)

                en_avg1, sd_avg1, ssim_avg1, cc_avg1, sf_avg1, vif_avg1 = save_result_and_evaluate(layers, result_path,
                                                                                                   log_tag[0] + '_' + models_unet[index],
                                                                                                   data[0], gen_unet[0],
                                                                                                   label[0], k)
                en_avg2, sd_avg2, ssim_avg2, cc_avg2, sf_avg2, vif_avg2 = save_result_and_evaluate(layers, result_path,
                                                                                                   log_tag[1] + '_' +
                                                                                                   models_15srgan_patch[index],
                                                                                                   data[0],
                                                                                                   gen_15srgan_patch[0],
                                                                                                   label[0], k)
                en_avg3, sd_avg3, ssim_avg3, cc_avg3, sf_avg3, vif_avg3 = save_result_and_evaluate(layers, result_path,
                                                                                                   log_tag[2] + '_' +
                                                                                                   models_unet_patch_64[index],
                                                                                                   data[0],
                                                                                                   gen_unet_patch_64[0],
                                                                                                   label[0], k)
                en_avg4, sd_avg4, ssim_avg4, cc_avg4, sf_avg4, vif_avg4 = save_result_and_evaluate(layers, result_path,
                                                                                                   log_tag[3] + '_' +
                                                                                                   models_unet_patch_32[index],
                                                                                                   data[0],
                                                                                                   gen_unet_patch_32[0],
                                                                                                   label[0], k)
                en_avg5, sd_avg5, ssim_avg5, cc_avg5, sf_avg5, vif_avg5 = save_result_and_evaluate(layers, result_path,
                                                                                                   log_tag[4] + '_' +
                                                                                                   models_unet_blur[index],
                                                                                                   data[0],
                                                                                                   gen_unet_blur[0],
                                                                                                   label[0], k)

                en_avg += np.array([en_avg1 , en_avg2 , en_avg3 , en_avg4 , en_avg5])
                sd_avg += np.array([sd_avg1 , sd_avg2 , sd_avg3 , sd_avg4 , sd_avg5])
                ssim_avg += np.array([ssim_avg1 , ssim_avg2 , ssim_avg3 , ssim_avg4 , ssim_avg5])
                cc_avg += np.array([cc_avg1 , cc_avg2 , cc_avg3 , cc_avg4 , cc_avg5])
                sf_avg += np.array([sf_avg1 , sf_avg2 , sf_avg3 , sf_avg4 , sf_avg5])
                vif_avg += np.array([vif_avg1 , vif_avg2 , vif_avg3 , vif_avg4 , vif_avg5])
            k += 1
        en_avg /= k
        sd_avg /= k
        ssim_avg /= k
        cc_avg /= k
        sf_avg /= k
        vif_avg /= k

        en_avgt.append(en_avg)
        sd_avgt.append(sd_avg)
        ssim_avgt.append(ssim_avg)
        cc_avgt.append(cc_avg)
        sf_avgt.append(sf_avg)
        vif_avgt.append(vif_avg)

        print(en_avg , sd_avg , ssim_avg , cc_avg , sf_avg , vif_avg)

    np.save('en' , np.array(en_avgt))
    np.save('sd', np.array(sd_avgt))
    np.save('ssim', np.array(ssim_avgt))
    np.save('cc', np.array(cc_avgt))
    np.save('sf', np.array(sf_avgt))
    np.save('vif', np.array(vif_avgt))


def test_imgs():
    layers = [0]
    img_num = 10

    datas , labels = read_imgs(layers , img_num = img_num)
    print(np.shape(datas) , np.shape(labels))

    model_path_unet = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_layer\\netG_epoch_8_5883.pth'
    model_path_15srgan_patch = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_layer_patch\\netG_epoch_7_4858.pth'
    model_path_unet_patch_64 = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_unet_patch64\\netG_epoch_8_5651.pth'
    model_path_unet_patch_32 = 'V:\\fusion_0_unet_patch32\\checkpoints_unet\\netG_epoch_8_5956.pth'
    model_path_unet_blur = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_layer_c\\netG_epoch_8_5958.pth'

    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_0_layer_test_all\\imgs\\'

    model_unet = Light(3 , 3)
    model_unet.load_state_dict(torch.load(model_path_unet))
    model_unet.cuda()

    model_15srgan_patch = SingleLayerGenerator(15)
    model_15srgan_patch.load_state_dict(torch.load(model_path_15srgan_patch))
    model_15srgan_patch.cuda()

    model_unet_patch_64 = Light(3 , 3)
    model_unet_patch_64.load_state_dict(torch.load(model_path_unet_patch_64))
    model_unet_patch_64.cuda()

    model_unet_patch_32 = Light(3, 3)
    model_unet_patch_32.load_state_dict(torch.load(model_path_unet_patch_32))
    model_unet_patch_32.cuda()

    model_unet_blur = Light(4, 3)
    model_unet_blur.load_state_dict(torch.load(model_path_unet_blur))
    model_unet_blur.cuda()

    k = 0

    en_avg , sd_avg , ssim_avg , cc_avg , sf_avg , vif_avg = [] , [] , [] , [] , [] , []
    for data , label in zip(datas , labels):
        data = [data]
        label = [label]
        data = torch.from_numpy(np.array(data))
        label = torch.from_numpy(np.array(label))
        print(np.shape(data))
        data = data.cuda()
        with torch.no_grad():
            gen_unet = model_unet(data)
            gen_15srgan_patch = model_15srgan_patch(data)
            gen_unet_patch_64 = model_unet_patch_64(data)
            gen_unet_patch_32 = model_unet_patch_32(data)

            blur_mask = model_blur_seg(data)
            blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
            blur_mask = blur_mask.reshape((blur_mask.size(0), 1, blur_mask.size(1), blur_mask.size(2)))
            data_blur = torch.cat([data, blur_mask], dim=1)
            gen_unet_blur = model_unet_blur(data_blur)

            en_avg1, sd_avg1, ssim_avg1, cc_avg1, sf_avg1, vif_avg1 = save_result_and_evaluate(layers, result_path,
                                                                                               log_tag[0],
                                                                                               data[0], gen_unet[0],
                                                                                               label[0], k)
            en_avg2, sd_avg2, ssim_avg2, cc_avg2, sf_avg2, vif_avg2 = save_result_and_evaluate(layers, result_path,
                                                                                               log_tag[1],
                                                                                               data[0],
                                                                                               gen_15srgan_patch[0],
                                                                                               label[0], k)
            en_avg3, sd_avg3, ssim_avg3, cc_avg3, sf_avg3, vif_avg3 = save_result_and_evaluate(layers, result_path,
                                                                                               log_tag[2],
                                                                                               data[0],
                                                                                               gen_unet_patch_64[0],
                                                                                               label[0], k)
            en_avg4, sd_avg4, ssim_avg4, cc_avg4, sf_avg4, vif_avg4 = save_result_and_evaluate(layers, result_path,
                                                                                               log_tag[3],
                                                                                               data[0],
                                                                                               gen_unet_patch_32[0],
                                                                                               label[0], k)
            en_avg5, sd_avg5, ssim_avg5, cc_avg5, sf_avg5, vif_avg5 = save_result_and_evaluate(layers, result_path,
                                                                                               log_tag[4],
                                                                                               data[0] ,
                                                                                               gen_unet_blur[0],
                                                                                               label[0], k)



            en_avg.append([en_avg1 , en_avg2 , en_avg3 , en_avg4 , en_avg5])
            sd_avg.append([sd_avg1 , sd_avg2 , sd_avg3 , sd_avg4 , sd_avg5])
            ssim_avg.append([ssim_avg1 , ssim_avg2 , ssim_avg3 , ssim_avg4 , ssim_avg5])
            cc_avg.append([cc_avg1 , cc_avg2 , cc_avg3 , cc_avg4 , cc_avg5])
            sf_avg.append([sf_avg1 , sf_avg2 , sf_avg3 , sf_avg4 , sf_avg5])
            vif_avg.append([vif_avg1 , vif_avg2 , vif_avg3 , vif_avg4 , vif_avg5])
        k += 1
    en_avg = np.array(en_avg)
    sd_avg = np.array(sd_avg)
    ssim_avg = np.array(ssim_avg)
    cc_avg = np.array(cc_avg)
    sf_avg = np.array(sf_avg)
    vif_avg = np.array(vif_avg)

    np.save('en1' , np.array(en_avg))
    np.save('sd1', np.array(sd_avg))
    np.save('ssim1', np.array(ssim_avg))
    np.save('cc1', np.array(cc_avg))
    np.save('sf1', np.array(sf_avg))
    np.save('vif1', np.array(vif_avg))


def draw_map():
    en = np.load('en1.npy')
    sd = np.load('sd1.npy')
    ssim = np.load('ssim1.npy')
    cc = np.load('cc1.npy')
    sf = np.load('sf1.npy')
    vif = np.load('vif1.npy')

    print(np.shape(en) , np.shape(sd) , np.shape(ssim) , np.shape(cc) , np.shape(sf) , np.shape(vif))

    x = [t for t in range(0 , 10)]

    plt.plot(x , sf[ : , 0 , 1], c=color[0], label = log_tag[0])
    plt.plot(x, sf[:, 1 , 1], c=color[1], label = log_tag[1])
    plt.plot(x , sf[: , 2 , 1] , c = color[2] , label = log_tag[2])
    plt.plot(x, sf[:, 3 , 1], c=color[3], label = log_tag[3])
    plt.plot(x , sf[: , 4 , 1] , c = color[4] , label = log_tag[4])
    plt.plot(x, sf[:, 4, 0], c=color[5], label=log_tag[5])
    if True:
        plt.plot(x, sf[:, 4 , 2], c=color[6], label=log_tag[6])
        # plt.plot(x, ssim[:, 0], c='magenta', label=tag)

    plt.legend(loc = 'best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("multi_imgs_sf")

    plt.show()


if __name__ == '__main__':
    # test_models()
    # test_imgs()
    draw_map()