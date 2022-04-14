import os
import cv2
import numpy as np
import torch
from compare_fusion.compare_models import MFICnn
from skimage import morphology
from scipy import ndimage as ndi
from compare_fusion.guided_filter.GuideFilter import guided_filter

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

img_size = 512
stride = 2
block_size = 16

# set weight path
cnn_model_path = 'X:\\GXB\\paper&patent\\cnn\\train_log\\cnn_model_epoch_99_214.pth'

cnn_model = MFICnn(1).cuda()  # load model
cnn_model.load_state_dict(torch.load(cnn_model_path))
cnn_model.cuda()


def fusion_imgs(img_a , img_b):
    r_img_a , r_img_b = img_a.copy() , img_b.copy()

    img_a = cv2.cvtColor(img_a , cv2.COLOR_BGR2GRAY)
    img_b = cv2.cvtColor(img_b , cv2.COLOR_BGR2GRAY)

    img_a, img_b = np.float32(img_a / 255.), np.float32(img_b / 255.)

    img_a, img_b = torch.from_numpy(img_a).cuda(), torch.from_numpy(img_b).cuda()
    img_a = torch.reshape(img_a, (1, 1) + img_a.size())
    img_b = torch.reshape(img_b, (1, 1) + img_b.size())

    gen_label = cnn_model(img_a, img_b, test=True)

    # map the mask to the img
    gen_label = gen_label.cpu().detach().numpy()[0, 0, :, :]
    mask = np.zeros((img_size, img_size), np.float32)

    for x in range(0, gen_label.shape[0]):
        for y in range(0, gen_label.shape[1]):
            mask[x * stride: (x * stride + block_size), y * stride: (y * stride + block_size)] += gen_label[x, y]
            if x != 0 and y == 0:
                mask[x * stride: (x * stride + block_size - stride), :] /= 2
            if y != 0 and x == 0:
                mask[:, y * stride: (y * stride + block_size - stride)] /= 2
            if y != 0 and x != 0:
                mask[x * stride: (x * stride + block_size), y * stride: (y * stride + block_size)] /= 2
                mask[(x * stride + block_size - 2): (x * stride + block_size),
                (y * stride + block_size - 2): (y * stride + block_size)] *= 2

    mask = mask > 0.5
    mask = morphology.remove_small_objects(mask, img_size * img_size * 0.01)
    mask = ndi.binary_fill_holes(mask)

    mask = guided_filter(np.uint8(mask * 255))
    mask = np.clip(mask, 0, 1)
    mask = np.stack([mask , mask , mask] , axis = 2)

    fusion_img = r_img_a * mask + r_img_b * (1 - mask)

    return np.uint8(fusion_img)

def fusion_img_by_cnn(path , cz , layers = [-1 , 0 , 1]):
    img = cv2.imread(path + cz + '_0.tif')

    for layer in range(0 , int(len(layers) / 2)):
        img1 = cv2.imread(path + cz + '_' + str(layers[layer]) + '.tif')
        img2 = cv2.imread(path + cz + '_' + str(layers[len(layers) - layer - 1]) + '.tif')
        temp = fusion_imgs(img1 , img2)
        img = fusion_imgs(img , temp)

    return img

if __name__ == '__main__':

    source_path1 = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares_new\\ori_-1-1_layer\\'
    source_path2 = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\ori_-2-2_layer\\'

    target_path1 = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares_new\\gen_-1-1_fusion\\CNN\\'
    target_path2 = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_-2-2_fusion\\CNN\\'

    with open('X:\\GXB\\20x_and_40x_data\\test_1.txt') as test:

        layers1 = [-1 , 0 , 1]
        layers2 = [-2 , -1 , 0 , 1 , 2]

        for line in test:

            line = line[ : line.find('.tif')]

            fusion_3_layers = fusion_img_by_cnn(source_path1 , line , layers1)
            # fusion_5_layers = fusion_img_by_cnn(source_path2 , line , layers2)

            cv2.imwrite(target_path1 + line + '.tif' , fusion_3_layers)
            # cv2.imwrite(target_path2 + line + '.tif', fusion_5_layers)


