from compare_fusion.compare_test.data_read_save import *
from model.model import Light
from compare_fusion.compare_models import Pixel2PixelGenerator
from compare_fusion.compare_models import FusionGanGenerator
from compare_fusion.compare_models import SRGANGenerator
from model.torch_model_resnet_aspp import ResNetASPP
import os
import torch
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# load blur model
mask_threshold = 0.5
model_path = '/mnt/68_gxb/blur_model/44_1125.pth'
model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
model_blur_seg.load_state_dict(torch.load(model_path))
model_blur_seg.cuda()

if __name__ == '__main__':

    # set test model path
    unet_catblur_patch64_model_path = '/mnt/sda2/20x_and_40x_data/train_log/our_blur_3_layer/checkpoints/netG_epoch_10_5286.pth'
    # unet_catblur_patch64_model_path_temp = 'V:\\fusion_-1-1_unet_catblur_patch64_new2\\checkpoints\\netG_epoch_4_3323.pth'
    pixel2pixel_model_path = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\pixel2pixel_-1-1_layer\\netG_epoch_2_182.pth'
    fusionGan_model_path = 'V:\\compare_fusion\\fusionGan_-1-1_layer\\checkpoints\\netG_epoch_3_441.pth'
    srgan_model_path = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\srgan_fusion_-1-1_layer\\netG_epoch_6_4068.pth'
    unet_patch64_model_path = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_-1-1_unet\\netG_epoch_3_1494.pth'

    # initial test model
    model_unet_patch64_catblur = Light(12, 3)
    model_unet_patch64_catblur.load_state_dict(torch.load(unet_catblur_patch64_model_path))
    model_unet_patch64_catblur.cuda()

    # model_pixel2pixel = Pixel2PixelGenerator(9 , 3)
    # model_pixel2pixel.load_state_dict(torch.load(pixel2pixel_model_path))
    # model_pixel2pixel.cuda()
    #
    # model_fusionGan = FusionGanGenerator(9 , 3)
    # model_fusionGan.load_state_dict(torch.load(fusionGan_model_path))
    # model_fusionGan.cuda()
    #
    # model_srgan = SRGANGenerator(9 , 5)
    # model_srgan.load_state_dict(torch.load(srgan_model_path))
    # model_srgan.cuda()
    #
    # model_unet_patch64 = Light(9 , 3)
    # model_unet_patch64.load_state_dict(torch.load(unet_patch64_model_path))
    # model_unet_patch64.cuda()

    path = '/mnt/sda2/20x_and_40x_data/split_data/'
    layers = [-1 , 0 , 1]
    imgs , names = read_data(path , layers)

    #unet_patch64_catblur predict
    # temp_imgs = imgs.copy() / 255.
    # temp_imgs = torch.from_numpy(temp_imgs)

    unet_patch64_catblur_gen_imgs = []
    with torch.no_grad():
        for i in tqdm(range(len(imgs))):
            temp_img = np.float32(imgs[i]) / 255.
            temp_img = torch.from_numpy(temp_img).cuda()
            img = torch.reshape(temp_img , (1 , ) + temp_img.size())
            blur_model_input = torch.reshape(temp_img , (len(layers) , 3 , temp_img.size(1) , temp_img.size(2)))
            blur_mask = model_blur_seg(blur_model_input)
            blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
            blur_mask = blur_mask.reshape((1 , ) + blur_mask.size())
            data_blur = torch.cat([img, blur_mask], dim=1)

            temp = model_unet_patch64_catblur(data_blur)
            temp = temp.cpu().detach().numpy()
            temp_img.cpu().detach()
            unet_patch64_catblur_gen_imgs.append(temp)
    unet_patch64_catblur_gen_imgs = np.concatenate(unet_patch64_catblur_gen_imgs , axis = 0)

    unet_patch64_catblur_path = '/mnt/sda2/20x_and_40x_data/train_log/compare/fusion_3_layer/our_BM/'
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
    # srgan_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_-1-1_fusion\\SRGAN\\'
    # save_data(srgan_path , srgan_gen_imgs, names, norm_way = 1)

    # unet_patch64_gen_imgs = []
    # with torch.no_grad():
    #     for i in tqdm(range(len(temp_imgs))):
    #         temp_img = temp_imgs[i].cuda()
    #         img = torch.reshape(temp_img , (1 , ) + temp_img.size())
    #         temp = model_unet_patch64(img)
    #         temp = temp.cpu().detach().numpy()
    #         unet_patch64_gen_imgs.append(temp)
    #     unet_patch64_gen_imgs = np.concatenate(unet_patch64_gen_imgs , axis = 0)
    #
    # unet_patch64_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_-1-1_fusion\\unet_patch64\\'
    # save_data(unet_patch64_path , unet_patch64_gen_imgs , names , norm_way = 1)

    #pixel2pixel predict
    # temp_imgs = imgs.copy() / 127.5 - 1
    # temp_imgs = torch.from_numpy(temp_imgs)
    #
    # pixel2pixel_gen_imgs = []
    # with torch.no_grad():
    #     for i in tqdm(range(len(temp_imgs))):
    #         temp_img = temp_imgs[i].cuda()
    #         img = torch.reshape(temp_img , (1,) + temp_img.size())
    #         temp = model_pixel2pixel(img)
    #         temp = temp.cpu().detach().numpy()
    #         temp_img.cpu().detach()
    #         pixel2pixel_gen_imgs.append(temp)
    # pixel2pixel_gen_imgs = np.concatenate(pixel2pixel_gen_imgs , axis = 0)
    # pixel2pixel_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_-1-1_fusion\\Pixel2Pixel\\'
    # save_data(pixel2pixel_path , pixel2pixel_gen_imgs , names , norm_way = 2)
    #
    # #fusionGan predict
    #
    # fusionGan_gen_imgs = []
    # with torch.no_grad():
    #     for i in tqdm(range(len(temp_imgs))):
    #         temp_img = temp_imgs[i].cuda()
    #         img = torch.reshape(temp_img, (1,) + temp_img.size())
    #         temp = model_fusionGan(img)
    #         temp = temp.cpu().detach().numpy()
    #         temp_img.cpu().detach()
    #         fusionGan_gen_imgs.append(temp)
    # fusionGan_gen_imgs = np.concatenate(fusionGan_gen_imgs , axis = 0)
    # fusionGan_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_-1-1_fusion\\FusionGan\\'
    # save_data(fusionGan_path , fusionGan_gen_imgs , names , norm_way = 2)