'''
    use unet train 8 epoch , -2 , 2 , 0-layer to fusion img
'''


import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from model import LightR
# from model import SingleLayerGenerator
from data.dataset import Fusion20xMultiLayersTripleDataSet
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.autograd import Variable
import cv2
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

path = '/mnt/zhaolu/GXB/20x_and_40x_data/'
image_20x_data_path = path + 'split_data/'
image_20x_label_path = path + 'label/'
names_log = path + 'train.txt'
num_workers = 0
shuffle_flag = True

##############
device = torch.device('cuda')

layers = [-1 , 0 , 1]
train_set = Fusion20xMultiLayersTripleDataSet(image_20x_data_path , image_20x_label_path , names_log , layers)
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=shuffle_flag,
                          num_workers=num_workers)  # 训练数据加载器
# load model , use deep net
netG = LightR(9 , 3).cuda()

if __name__ == "__main__":

        for i, (img20x_data , img20x_label) in enumerate(train_loader):
            ############################
            # (1) Update G
            ###########################
            img20x_data = img20x_data.to(device)
            img20x_label = img20x_label.to(device)

            gen20x_label = netG(img20x_data)  # 生成器生成10x和20x的图像

            if i % img_log_step == 0:
                img20x_data  = img20x_data[0].cpu().detach().numpy()
                img20x_label = img20x_label[0].cpu().detach().numpy()
                gen20x_label = gen20x_label[0].cpu().detach().numpy()
                print(np.shape(img20x_data) , np.shape(img20x_label) , np.shape(gen20x_label))
                temp = []
                for k in range(0 , len(layers)):
                    temp.append(np.transpose(img20x_data[k * 3 : (k + 1) * 3] , [1 , 2 , 0]) * 255)
                temp.append(np.transpose(gen20x_label , [1 , 2 , 0]) * 255)
                temp.append(np.transpose(img20x_label , [1 , 2 , 0]) * 255)
                img20x_data = cv2.hconcat(np.uint8(temp))
                print(np.shape(img20x_data))
                img20x_data = cv2.cvtColor(img20x_data , cv2.COLOR_RGB2BGR)

                cv2.imwrite(image_log_path + 'stage1_' + str(1) + '_' + str(image_save_counter) + '.tif', img20x_data)