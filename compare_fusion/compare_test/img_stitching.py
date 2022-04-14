from compare_fusion.compare_test.data_read_save import *
from model.model import Light
from model.torch_model_resnet_aspp import ResNetASPP
import os
import numpy as np
import openslide
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

save_path = 'X:\\GXB\\paper&patent\\impl\\-1-1_fusion\\'

mask_threshold = 0.5

#定义读取的位置以及读取的大小
position = (40892 , 47676)
size = (6656 , 6656)

block_size = 512 #定义patch的大小
ratio = 1 / 4 #定义冗余区域


def read_img_3(path , cz , layers = [-1 , 0 , 1]):

    imgs = []
    for layer in layers:
        ors = openslide.OpenSlide(path + cz + '_Wholeslide_' + str(layer) + '.tif')
        img = ors.read_region(position, 0, size)
        ors.close()
        img = np.array(img)
        img = img[: , : , 0 : 3]
        img = img[: , : , ::-1]
        cv2.imwrite(save_path + cz + '_' + str(layer) + '.tif' , img)
        imgs.append(img)
    imgs = np.array(imgs)

    split_imgs = []
    position_list = []

    w_nums = (size[0] - block_size) // int((1 - ratio) * block_size) + 1
    h_nums = (size[1] - block_size) // int((1 - ratio) * block_size) + 1


    for w_i in range(0 , w_nums):
        for h_i in range(0 , h_nums):
            move_dis = int((1 - ratio) * block_size)
            position_list.append([w_i * move_dis , h_i * move_dis])
            temps = []
            for layer in layers:
                temp = imgs[layer , h_i * move_dis : h_i * move_dis + block_size , w_i * move_dis : w_i * move_dis + block_size , :]
                temp = cv2.cvtColor(temp , cv2.COLOR_BGR2RGB)
                temp = np.float32(temp) / 255.
                temps.append(temp)
            temp = np.concatenate(temps , axis = 2)
            temp = np.transpose(temp , [2 , 0 , 1])
            split_imgs.append(temp)

    return split_imgs , position_list

#进行图像块的读取与拆分
def read_img(path , cz , layers = [0]):

    img_path = path + cz + '_Wholeslide_' + str(layers[0]) + '.tif'
    print(img_path)
    ors = openslide.OpenSlide(img_path)
    img = ors.read_region(position , 0 , size)
    ors.close()
    img = np.array(img)
    img = img[: , : , 0 : 3]
    img = img[: , : , ::-1]
    cv2.imwrite(save_path + cz + '_0.tif' , img)

    ors = openslide.OpenSlide(path + cz + '_Wholeslide_Extended.tif')
    label_img = ors.read_region(position , 0 , size)
    ors.close()
    label_img = np.array(label_img)
    label_img = label_img[: , : , 0 : 3]
    label_img = label_img[: , : , ::-1]
    cv2.imwrite(save_path + cz + '_l.tif' , label_img)

    split_imgs = []
    position_list = []

    w_nums = (size[0] - block_size) // int((1 - ratio) * block_size) + 1
    h_nums = (size[1] - block_size) // int((1 - ratio) * block_size) + 1

    for w_i in range(0 , w_nums):
        for h_i in range(0 , h_nums):
            move_dis = int((1 - ratio) * block_size)
            position_list.append([w_i * move_dis , h_i * move_dis])
            temp = img[h_i * move_dis : h_i * move_dis + block_size , w_i * move_dis : w_i * move_dis + block_size , :]
            temp = cv2.cvtColor(temp , cv2.COLOR_BGR2RGB)
            temp = np.float32(temp) / 255.
            temp = np.transpose(temp , [2 , 0 , 1])
            split_imgs.append(temp)

    return split_imgs , position_list

#进行图像块的拼接与保存
def stitch(split_results , position_list , cz):

    result_img = np.zeros(size + (3 , ) , dtype = np.uint8)

    for i in range(len(position_list)):
        temp = np.uint8(split_results[i] * 255)
        temp = np.transpose(temp , [1 , 2 , 0])
        temp = cv2.cvtColor(temp , cv2.COLOR_RGB2BGR)

        redu_region = int(block_size * ratio)
        #处理 w 方向上的重叠区域
        if position_list[i][0] != 0:
            left_region = result_img[position_list[i][1] : position_list[i][1] + block_size , position_list[i][0] : position_list[i][0] + redu_region]
            right_region = temp[: , 0 : redu_region]
            # f_region = np.zeros(right_region.shape , dtype = np.uint8)
            # for w in range(redu_region):
            #     for h in range(0 , 512):
            #         f_region[h , w] = left_region[h , w] * ((redu_region - w) / redu_region) + \
            #                           right_region[h , w] * (w / redu_region)
            f_region = np.uint8((np.float32(left_region) + np.float32(right_region)) / 2)
            result_img[position_list[i][1]: position_list[i][1] + block_size,
            position_list[i][0] : position_list[i][0] + redu_region] = f_region
        else:
            result_img[position_list[i][1]: position_list[i][1] + block_size,
            position_list[i][0] : position_list[i][0] + redu_region] = temp[: , 0 : redu_region]

        #处理 h 方向上的重叠区域
        if position_list[i][1] != 0:
            up_region = result_img[position_list[i][1] : position_list[i][1] + redu_region , position_list[i][0] : position_list[i][0] + block_size]
            down_region = temp[0 : redu_region , :]
            # f_region = np.zeros(up_region.shape , dtype = np.uint8)
            # for h in range(redu_region):
            #     for w in range(0 , 512):
            #         f_region[h , w] = up_region[h , w] * ((redu_region - h) / redu_region) + \
            #                           down_region[h , w] * (h / redu_region)
            f_region = np.uint8((np.float32(up_region) + np.float32(down_region)) / 2)
            result_img[position_list[i][1]: position_list[i][1] + redu_region,
            position_list[i][0]: position_list[i][0] + block_size] = f_region
        else:
            result_img[position_list[i][1]: position_list[i][1] + redu_region ,
            position_list[i][0] : position_list[i][0] + block_size] = temp[0 : redu_region , :]

        result_img[position_list[i][1] + redu_region : position_list[i][1] + block_size ,
        position_list[i][0] + redu_region : position_list[i][0] + block_size] = temp[redu_region : , redu_region : ]

    cv2.imwrite(save_path + cz + '_r.tif' , result_img)


#进行模型的初始化
def init_model():
    # load blur model
    model_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\middle_result\\weights_log\\44_1125.pth'
    model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
    model_blur_seg.load_state_dict(torch.load(model_path))
    model_blur_seg.cuda()

    unet_catblur_patch64_model_path = 'V:\\fusion_-1-1_unet_catblur_patch64_new2\\checkpoints\\netG_epoch_4_3323.pth'

    # initial test model
    model_unet_patch64_catblur = Light(12 , 3)
    model_unet_patch64_catblur.load_state_dict(torch.load(unet_catblur_patch64_model_path))
    model_unet_patch64_catblur.cuda()

    return model_unet_patch64_catblur , model_blur_seg

if __name__ == '__main__':
    path = 'X:\\TCT\\20x_and_40x_new\\20x_tiff\\'
    czs = os.listdir(path)

    # 0 layer fusion
    # model_unet_patch64_catblur, model_blur_seg = init_model()
    # for cz in czs:
    #
    #     split_imgs , position_list = read_img(path + cz + '\\' , cz)
    #     print(np.shape(split_imgs))
    #     split_imgs = torch.from_numpy(np.array(split_imgs)).cuda()
    #     split_results = []
    #     with torch.no_grad():
    #         for i in tqdm(range(len(split_imgs))):
    #             img = torch.reshape(split_imgs[i] , (1 , ) + split_imgs[i].size())
    #             blur_mask = model_blur_seg(img)
    #             blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
    #             blur_mask = blur_mask.reshape((blur_mask.size(0), 1, blur_mask.size(1), blur_mask.size(2)))
    #             data_blur = torch.cat([img, blur_mask], dim=1)
    #
    #             temp = model_unet_patch64_catblur(data_blur)
    #             temp = temp.cpu().detach().numpy()
    #             split_results.append(temp[0])
    #     split_imgs.cpu().detach().numpy()
    #     stitch(split_results , position_list , cz)

    # -1-1 layers fusion
    model_unet_patch64_catblur, model_blur_seg = init_model()
    layers = [-1 , 0 , 1]
    for cz in czs:
        split_results = []
        split_imgs, position_list = read_img_3(path + cz + '\\', cz)
        split_imgs = torch.from_numpy(np.array(split_imgs)).cuda()
        with torch.no_grad():
            for i in tqdm(range(len(split_imgs))):
                temp_img = split_imgs[i].cuda()
                img = torch.reshape(temp_img , (1 , ) + temp_img.size())
                blur_model_input = torch.reshape(temp_img , (len(layers) , 3 , temp_img.size(1) , temp_img.size(2)))
                blur_mask = model_blur_seg(blur_model_input)
                blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()
                blur_mask = blur_mask.reshape((1 , ) + blur_mask.size())
                data_blur = torch.cat([img, blur_mask], dim=1)

                temp = model_unet_patch64_catblur(data_blur)
                temp = temp.cpu().detach().numpy()
                temp_img.cpu().detach()
                split_results.append(temp[0])
            split_imgs.cpu().detach().numpy()
            stitch(split_results , position_list , cz)
