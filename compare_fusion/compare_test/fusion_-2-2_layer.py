from compare_fusion.compare_test.data_read_save import *
from model.model import Light
from compare_fusion.compare_models import Pixel2PixelGenerator
from compare_fusion.compare_models import FusionGanGenerator
from compare_fusion.compare_models import SRGANGenerator
from model.torch_model_resnet_aspp import ResNetASPP
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load blur model
mask_threshold = 0.5
model_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\middle_result\\weights_log\\44_1125.pth'
model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
model_blur_seg.load_state_dict(torch.load(model_path))
model_blur_seg.cuda()

if __name__ == '__main__':

    # set test model path
    unet_catblur_patch64_model_path = 'V:\\fusion_-2-2_unet_catblur_patch64\\checkpoints\\netG_epoch_0_745.pth'
    pixel2pixel_model_path = 'V:\\compare_fusion\\pixel2pixel_-2-2_layer\\checkpoints\\netG_epoch_1_86.pth'
    fusionGan_model_path = 'V:\\compare_fusion\\fusionGan_-2-2_layer\\checkpoints\\netG_epoch_4_636.pth'
    srgan_model_path = 'X:\GXB\\20x_and_40x_data\\checkpoints\\srgan_fusion_-2-2_layer\\netG_epoch_2_1108.pth'

    # initial test model
    # initial test model
    # model_unet_patch64_catblur = Light(20, 3)
    # model_unet_patch64_catblur.load_state_dict(torch.load(unet_catblur_patch64_model_path))
    # model_unet_patch64_catblur.cuda()
    #
    # model_pixel2pixel = Pixel2PixelGenerator(15 , 3)
    # model_pixel2pixel.load_state_dict(torch.load(pixel2pixel_model_path))
    # model_pixel2pixel.cuda()
    #
    # model_fusionGan = FusionGanGenerator(15 , 3)
    # model_fusionGan.load_state_dict(torch.load(fusionGan_model_path))
    # model_fusionGan.cuda()

    # model_srgan = SRGANGenerator(15 , 5)
    # model_srgan.load_state_dict(torch.load(srgan_model_path))
    # model_srgan.cuda()

    path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\ori_-2-2_layer\\'
    layers = [-2 , -1 , 0 , 1 , 2]
    imgs , names = read_data(path , layers)

    imgs = np.float32(imgs)

    #unet_patch64_catblur predict
    temp_imgs = imgs.copy() / 255.
    temp_imgs = torch.from_numpy(temp_imgs)
    #
    unet_patch64_catblur_gen_imgs = []
    with torch.no_grad():
        for i in tqdm(range(len(temp_imgs))):
            temp_img = temp_imgs[i].cuda()
            img = torch.reshape(temp_img , (1 , ) + temp_img.size())
            blur_model_input = torch.reshape(temp_img , (len(layers) , 3 , temp_img.size(1) , temp_img.size(2)))
            blur_mask = model_blur_seg(blur_model_input)

            print(np.shape(blur_mask))

    #         blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
    #         blur_mask = blur_mask.reshape((1 , ) + blur_mask.size())
    #         data_blur = torch.cat([img, blur_mask], dim=1)
    #
    #         temp = model_unet_patch64_catblur(data_blur)
    #         temp = temp.cpu().detach().numpy()
    #         temp_img.cpu().detach()
    #         unet_patch64_catblur_gen_imgs.append(temp)
    # unet_patch64_catblur_gen_imgs = np.concatenate(unet_patch64_catblur_gen_imgs , axis = 0)
    #
    # unet_patch64_catblur_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_-2-2_fusion\\unet_catblur_patch64\\'
    # save_data(unet_patch64_catblur_path , unet_patch64_catblur_gen_imgs , names , norm_way = 1)

    # srgan_gen_imgs = []
    # with torch.no_grad():
    #     for i in tqdm(range(len(temp_imgs))):
    #         temp_img = temp_imgs[i].cuda()
    #         img = torch.reshape(temp_img , (1 , ) + temp_img.size())
    #         temp = model_srgan(img)
    #         temp = temp.cpu().detach().numpy()
    #         srgan_gen_imgs.append(temp)
    # srgan_gen_imgs = np.concatenate(srgan_gen_imgs , axis = 0)
    # srgan_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_-2-2_fusion\\SRGAN\\'
    # save_data(srgan_path , srgan_gen_imgs, names, norm_way = 1)

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
    # pixel2pixel_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_-2-2_fusion\\Pixel2Pixel\\'
    # save_data(pixel2pixel_path , pixel2pixel_gen_imgs , names , norm_way = 2)

    #fusionGan predict

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
    # fusionGan_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_-2-2_fusion\\FusionGan\\'
    # save_data(fusionGan_path , fusionGan_gen_imgs , names , norm_way = 2)
