"""
    caculate the gusion img metrics
"""

from tqdm import tqdm
import cv2
import os
import numpy as np
from fusion_metrics.metrics import Metrics
from multiprocessing import cpu_count
from multiprocessing import dummy

me = Metrics() #define the class of metrics

def cal_metrics(img_a , img_b , channel = 1):

    ssim = me.ssim_m(img_a , img_b , multichannel = (channel == 3))
    cc = me.correlation_coe(img_a , img_b , channel = channel)
    nmi = me.nmi(img_a , img_b , channel = channel) # normalized mutual info score

    return {'ssim' : ssim , 'cc' : cc , 'nmi' : nmi}

gen_0_fusion = ['FusionGAN' , 'Pixel2Pixel' , 'SRGAN' , 'our_BM' , 'our_no_BM']
gen_m_fusion = ['FusionGAN' , 'Pixel2Pixel' , 'SRGAN' , 'our_BM' , 'WT' , 'DTCWT' ,
                'LPP' , 'CNN' , 'our_no_BM']

# 计算测试的100张图片的指标
def me_k():
    path = r'X:\GXB\20x_and_40x_data\test_result\compares_new\temp'
    label_path = r'X:\GXB\20x_and_40x_data\test_result\compares_new\label'
    items = os.listdir(path)
    items = [x for x in items if x.find('.txt') == -1]
    print(items)
    for item in items[6 : 7]:

        f = open(path + '/' + item + '.txt' , 'w')

        names = os.listdir(path + '/' + item)
        names = [x for x in names if x.find('.tif') != -1]
        for name in tqdm(names):
            img_a = cv2.imread(path + '/' + item + '/' + name , 0)
            label = cv2.imread(label_path + '/' + name , 0)
            re = cal_metrics(img_a , label)
            f.write(name + '\t' + str(re['ssim']) + '\t' + str(re['cc']) + '\t' + str(re['nmi']) + '\n')

        f.close()

def cl_me():

    path = r'X:\GXB\20x_and_40x_data\test_result\compares_new\temp'
    label_path = r'X:\GXB\20x_and_40x_data\test_result\compares_new\label'
    items = os.listdir(path)
    items = [x for x in items if x.find('.txt') != -1]
    print(items)
    for item in items:
        ssim , cc , nmi = [] , [] , []
        with open(path + '/' + item) as f:

            for line in f:
                line = line[ : -1]
                line_uints = line.split('\t')
                ssim.append(np.float32(line_uints[1]))
                cc.append(np.float32(line_uints[2]))
                nmi.append(np.float32(line_uints[3]))

        ssim , cc , nmi = np.array(ssim) , np.array(cc) , np.array(nmi)

        print(item , np.mean(ssim) , np.mean(cc) , np.mean(nmi))


if __name__ == '__main__':

    cl_me()

    # me_k()

    # outside_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares_new\\'
    #
    # lines = []
    # with open('X:\\GXB\\20x_and_40x_data\\test_1.txt') as test:
    #     for line in tqdm(test):
    #         line = line[: line.find('.tif')]
    #         lines.append(line)
    #
    # cs = 'gen_0_fusion/our_no_BM'
    # f = open(outside_path + cs + '_metrics.txt', 'w')
    # for line in tqdm(lines):
    #     img_a = cv2.imread(outside_path + cs + '/' + line + '.tif' , 0)
    #     label = cv2.imread(outside_path + 'label/' + line + '.tif' , 0)
    #     re = cal_metrics(img_a , label)
    #     f.write(line + '\t' + str(re['ssim']) + '\t' + str(re['cc']) + '\t' + str(re['nmi']) + '\n')
    #
    # f.close()
