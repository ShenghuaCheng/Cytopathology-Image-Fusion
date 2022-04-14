import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import cv2
from model.torch_model_resnet_aspp import ResNetASPP
from train_and_test.public_code import *
from model.model import *
import matplotlib.pyplot as plt
import openslide


# #对于模糊识别模型，需要预先加载进来
# mask_threshold = 0.5
# model_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\middle_result\\weights_log\\44_1125.pth'
# model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
# model_blur_seg.load_state_dict(torch.load(model_path))
# model_blur_seg.cuda()
#
#
# color = ['red', 'blue', 'green', 'black', 'cyan' , 'magenta' , 'yellow']
# log_tag = ['unet', '15srgan_patch16', 'unet_patch64', 'unet_patch32' , 'unet_blur' , '0_layer', '11_fusion']
#
# def save_result_and_evaluate(layers , result_path , weight , data , gen_label , label , img_k):
#
#
#     img20x_data = data.cpu().detach().numpy()
#     img20x_label = label.cpu().detach().numpy()
#     gen20x_label = gen_label.cpu().detach().numpy()
#     temp = []
#     for k in range(0, len(layers)):
#         temp.append(np.transpose(img20x_data[k * 3: (k + 1) * 3], [1, 2, 0]) * 255)
#
#     temp.append(np.transpose(gen20x_label, [1, 2, 0]) * 255)
#     temp.append(np.transpose(img20x_label, [1, 2, 0]) * 255)
#     img20x_data = cv2.hconcat(np.uint8(temp))
#     img20x_data = cv2.cvtColor(img20x_data, cv2.COLOR_RGB2BGR)
#
#     layer_0 = int(len(layers) / 2)
#     len_layers = len(layers)
#     img20x_layer_0 = img20x_data[: , layer_0 * 512 : (layer_0 + 1) * 512 , :]
#
#     gen20x_label = img20x_data[: , len_layers * 512 : (len_layers + 1) * 512 , :]
#     img20x_label = img20x_data[:, (len_layers + 1) * 512: (len_layers + 2) * 512, :]
#
#
#
#     ssim_f = metrics.ssim_m(img20x_label , gen20x_label)
#     ssim_0 = metrics.ssim_m(img20x_label , img20x_layer_0)
#
#     cc_f = metrics.correlation_coe(img20x_label , gen20x_label , channel = 3)
#     cc_0 = metrics.correlation_coe(img20x_label , img20x_layer_0 , channel = 3)
#
#     return [[ssim_0 , ssim_f] , [cc_0 , cc_f]]
#
#
#
# def test_imgs():
#     layers = [0]
#     img_num = 10
#
#     datas , labels = read_imgs(layers , img_num = img_num)
#     print(np.shape(datas) , np.shape(labels))
#
#     model_path_unet_patch_64 = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_0_unet_patch64\\netG_epoch_8_5651.pth'
#
#     result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_0_layer_test_all\\imgs\\'
#
#
#     model_unet_patch_64 = Light(3 , 3)
#     model_unet_patch_64.load_state_dict(torch.load(model_path_unet_patch_64))
#     model_unet_patch_64.cuda()
#
#     k = 0
#
#     en_avg , sd_avg , ssim_avg , cc_avg , sf_avg , vif_avg = [] , [] , [] , [] , [] , []
#     for data , label in zip(datas , labels):
#         data = [data]
#         label = [label]
#         data = torch.from_numpy(np.array(data))
#         label = torch.from_numpy(np.array(label))
#         print(np.shape(data))
#         data = data.cuda()
#         with torch.no_grad():
#             gen_unet_patch_64 = model_unet_patch_64(data)
#
#             ssim_avg3, cc_avg3 = save_result_and_evaluate(layers, result_path,
#                                                                              log_tag[2],
#                                                                               data[0],
#                                                                                gen_unet_patch_64[0],
#                                                                             label[0], k)
#             ssim_avg.append(ssim_avg3)
#             cc_avg.append(cc_avg3)
#         k += 1
#
#     ssim_avg = np.array(ssim_avg)
#     cc_avg = np.array(cc_avg)
#
#     np.save('ssim1', np.array(ssim_avg))
#     np.save('cc1', np.array(cc_avg))
#
#
# def draw_map():
#
#     ssim = np.load('ssim1.npy')
#     cc = np.load('cc1.npy')
#
#     x = [t for t in range(0 , 10)]
#     print(np.shape(cc))
#     plt.plot(x , ssim[ : , 0 , 1], c=color[0], label = log_tag[2])
#     plt.plot(x, ssim[:, 0 , 0], c=color[2], label=log_tag[5])
#     if False:
#         plt.plot(x, cc[:, 2], c=color[6], label=log_tag[6])
#         # plt.plot(x, ssim[:, 0], c='magenta', label=tag)
#
#     plt.legend(loc = 'best')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title("multi_imgs_ssim")
#
#     plt.show()


def generate_sample_from3d1():
    mrxses_path = 'X:\\TCT\\TCTDATA\\Shengfuyou_1th\\Positive\\'
    czs = os.listdir(mrxses_path)
    czs = [x for x in czs if x.find('.mrxs') != -1]

    for i in range(0 , len(czs)):
        ors = openslide.OpenSlide(mrxses_path + czs[i])
        print(ors.level_dimensions[0])

if __name__ == '__main__':
    # test_models()
    # test_imgs()
    # draw_map()
    generate_sample_from3d1()