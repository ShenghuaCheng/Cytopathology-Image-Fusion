from model.torch_model_resnet_aspp import ResNetASPP
import torch
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

mask_threshold = 0.5
model_path = '/mnt/68_gxb/blur_model/44_1125.pth'
model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
model_blur_seg.load_state_dict(torch.load(model_path))
model_blur_seg.cuda()

layers = [-1 , 0 , 1]
path = '/mnt/sda2/20x_and_40x_data/split_data/'

czs = ['10140015_10000_20352']

imgs = []
names = []
for cz in czs:
    for layer in layers:
        names.append(cz + '_' + str(layer))
        imgs.append(cv2.imread(path + cz + '_' + str(layer) + '.tif'))

imgs = (np.float32(imgs) / 255. - 0.5) * 2
imgs = np.transpose(imgs , [0 , 3 , 1 , 2])

imgs = torch.from_numpy(imgs).cuda()
with torch.no_grad():
    masks = model_blur_seg(imgs)

save_path = '/mnt/sda2/paper/imgs'

for img , mask , name in zip(imgs , masks , names):
    img = img.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()

    img = np.transpose(np.uint8((img / 2 + 0.5) * 255) , [1 , 2 , 0])
    mask = mask[0 , : , :] > 0.5

    mask = ndi.binary_fill_holes(mask)
    thre_vol = 900
    mask = morphology.remove_small_objects(mask , min_size=thre_vol)

    contours , _ = cv2.findContours(np.uint8(mask * 255) , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    img_contour = img.copy()
    cv2.drawContours(img_contour , contours , -1 , (0 , 0 , 255) , 2)

    cv2.imwrite(save_path + name + '_i.tif' , img)
    cv2.imwrite(save_path + name + '_c.tif', img_contour)
    cv2.imwrite(save_path + name + '_m.tif', np.uint8(mask * 255))




