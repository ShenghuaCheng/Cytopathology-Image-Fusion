from compare_fusion.compare_test.data_read_save import *
from model.model import Light
from compare_fusion.compare_models import Pixel2PixelGenerator
from compare_fusion.compare_models import FusionGanGenerator
from compare_fusion.compare_models import SRGANGenerator
from model.torch_model_resnet_aspp import ResNetASPP
import os
import torch
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load blur model
mask_threshold = 0.5
model_path = 'X:\\GXB\\20x_and_40x_data\\blur_task\\train_data\\train_second\\middle_result\\weights_log\\44_1125.pth'
model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
model_blur_seg.load_state_dict(torch.load(model_path))
model_blur_seg.cuda()

if __name__ == '__main__':

    # set test model path
    unet_catblur_patch64_model_path = r'D:\20x_and_40x_data\train_log\our_blur_0_layer\checkpoints\netG_epoch_10_5281.pth'
    pixel2pixel_model_path = r'V:\train_new\pixel2pixel_0_layer\checkpoints\netG_epoch_6_389.pth'
    fusionGan_model_path = 'V:\\compare_fusion\\fusionGan_0_layer\\checkpoints\\netG_epoch_4_617.pth'
    srgan_model_path = 'V:\\compare_fusion\\srgan_0_layer\\checkpoints\\netG_epoch_3_2030.pth'
    unet_patch64_model_path = 'V:\\fusion_0_unet_patch64\\checkpoints_unet\\netG_epoch_8_5906.pth'

    # initial test model
    model_unet_patch64_catblur = Light(4, 3)
    model_unet_patch64_catblur.load_state_dict(torch.load(unet_catblur_patch64_model_path))
    model_unet_patch64_catblur.cuda()

    # model_pixel2pixel = Pixel2PixelGenerator(3 , 3)
    # model_pixel2pixel.load_state_dict(torch.load(pixel2pixel_model_path))
    # model_pixel2pixel.cuda()
    #
    # model_fusionGan = FusionGanGenerator(3 , 3)
    # model_fusionGan.load_state_dict(torch.load(fusionGan_model_path))
    # model_fusionGan.cuda()
    #
    # model_srgan = SRGANGenerator(3 , 5)
    # model_srgan.load_state_dict(torch.load(srgan_model_path))
    # model_srgan.cuda()
    #
    # model_unet_patch64 = Light(3 , 3)
    # model_unet_patch64.load_state_dict(torch.load(unet_patch64_model_path))
    # model_unet_patch64.cuda()

    path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares_new\\ori_-1-1_layer\\'
    layers = [0]
    imgs , names = read_data(path , layers)

    imgs = np.float32(imgs)

    # unet_patch64_catblur predict
    temp_imgs = imgs.copy() / 255.
    temp_imgs = torch.from_numpy(temp_imgs)

    # get the result of unet_patch64
    # unet_patch64_gen_imgs = []
    # with torch.no_grad():
    #     for i in tqdm(range(len(temp_imgs))):
    #         img = torch.reshape(temp_imgs[i], (1,) + temp_imgs[i].size())
    #         temp = model_unet_patch64(img)
    #         temp = temp.cpu().detach().numpy()
    #         unet_patch64_gen_imgs.append(temp)
    # temp_imgs.cpu().detach().numpy()
    # unet_patch64_gen_imgs = np.concatenate(unet_patch64_gen_imgs , axis = 0)
    # unet_patch64_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_0_fusion\\unet_patch64\\'
    # save_data(unet_patch64_path , unet_patch64_gen_imgs , names , norm_way = 1)

    #
    unet_patch64_catblur_gen_imgs = []
    with torch.no_grad():
        for i in tqdm(range(len(temp_imgs))):
            img = torch.reshape(temp_imgs[i] , (1 , ) + temp_imgs[i].size())
            img = img.cuda()
            blur_mask = model_blur_seg(img)
            blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
            blur_mask = blur_mask.reshape((blur_mask.size(0), 1, blur_mask.size(1), blur_mask.size(2)))
            data_blur = torch.cat([img, blur_mask], dim=1)
            temp = model_unet_patch64_catblur(data_blur)
            img = img.cpu().detach()
            temp = temp.cpu().detach().numpy()
            unet_patch64_catblur_gen_imgs.append(temp)
    temp_imgs.cpu().detach().numpy()
    unet_patch64_catblur_gen_imgs = np.concatenate(unet_patch64_catblur_gen_imgs , axis = 0)

    unet_patch64_catblur_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares_new\\gen_0_fusion\\our_BM\\'
    save_data(unet_patch64_catblur_path , unet_patch64_catblur_gen_imgs , names , norm_way = 1)


    # srgan_gen_imgs = []
    # with torch.no_grad():
    #     for i in tqdm(range(len(temp_imgs))):
    #         temp_img = temp_imgs[i].cuda()
    #         img = torch.reshape(temp_img , (1 , ) + temp_img.size())
    #         temp = model_srgan(img)
    #         temp = temp.cpu().detach().numpy()
    #         srgan_gen_imgs.append(temp)
    # srgan_gen_imgs = np.concatenate(srgan_gen_imgs , axis = 0)
    # srgan_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_0_fusion\\SRGAN\\'
    # save_data(srgan_path , srgan_gen_imgs, names, norm_way = 1)

    #pixel2pixel predict
    # temp_imgs = imgs.copy() / 127.5 - 1
    # temp_imgs = torch.from_numpy(temp_imgs)
    #
    # pixel2pixel_gen_imgs = []
    # with torch.no_grad():
    #     for i in tqdm(range(len(temp_imgs))):
    #         img = torch.reshape(temp_imgs[i] , (1,) + temp_imgs[i].size())
    #         img = img.cuda()
    #         temp = model_pixel2pixel(img)
    #         img = img.cpu().detach()
    #         temp = temp.cpu().detach().numpy()
    #         pixel2pixel_gen_imgs.append(temp)
    # temp_imgs.cpu().detach().numpy()
    # pixel2pixel_gen_imgs = np.concatenate(pixel2pixel_gen_imgs , axis = 0)
    # pixel2pixel_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares_new\\gen_0_fusion\\Pixel2Pixel\\'
    # save_data(pixel2pixel_path , pixel2pixel_gen_imgs , names , norm_way = 2)
    #
    # #fusionGan predict
    #
    # fusionGan_gen_imgs = []
    # with torch.no_grad():
    #     for i in tqdm(range(len(temp_imgs))):
    #         img = torch.reshape(temp_imgs[i], (1,) + temp_imgs[i].size())
    #         temp = model_fusionGan(img)
    #         temp = temp.cpu().detach().numpy()
    #         fusionGan_gen_imgs.append(temp)
    # temp_imgs.cpu().detach().numpy()
    # fusionGan_gen_imgs = np.concatenate(fusionGan_gen_imgs , axis = 0)
    # fusionGan_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_0_fusion\\FusionGan\\'
    # save_data(fusionGan_path , fusionGan_gen_imgs , names , norm_way = 2)
