import cv2
import numpy as np
import os
import multiprocessing.dummy as multiprocessing
from multiprocessing import cpu_count
from fusion_metrics.metrics import Metrics

metrics = Metrics()

def get_item(layers , img_name , data_path , label_path , dtcwt_fusion_path = None):
    img20x_data = []
    for layer in layers:
        temp = cv2.imread(data_path + img_name + '_' + str(layer) + '.tif')
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

        #   读入图片的归一化 ， 两种不同归一化的方式
        # temp = np.transpose(temp, axes=(2, 0, 1)).astype(np.float32) / 255.
        temp = np.transpose(temp , axes = (2 , 0 , 1)).astype(np.float32) / 127.5 - 1
        img20x_data.append(temp)
    img20x_data = np.concatenate(img20x_data, axis=0)

    img_20x_label_path = os.path.join(label_path, img_name + '.tif')
    img_20x_label = cv2.imread(img_20x_label_path)

    if dtcwt_fusion_path != None:
        dtcwt_fusion_img = cv2.imread(dtcwt_fusion_path + img_name + '.tif')
        dtcwt_fusion_img = dtcwt_fusion_img[: , 512 : 1024 , :]

    # BGR to RGB
    img_20x_label = cv2.cvtColor(img_20x_label, cv2.COLOR_BGR2RGB)
    # H*W*C to C*H*W
    # img_20x_label = np.transpose(img_20x_label, axes=(2, 0, 1)).astype(np.float32) / 255.
    img_20x_label = np.transpose(img_20x_label, axes=(2, 0, 1)).astype(np.float32) / 127.5 - 1
    if dtcwt_fusion_path == None:
        return [img20x_data , img_20x_label]
    else:
        return [img20x_data , img_20x_label , dtcwt_fusion_img]

def read_imgs(layers , img_num = 1000 , dtcwt_fusion_path = None):
    data_path = 'X:\\GXB\\20x_and_40x_data\\split_data\\'
    label_path = 'X:\\GXB\\20x_and_40x_data\\label\\'

    pool = multiprocessing.Pool(cpu_count())
    data , label = [] , []
    dtctw_fusion_imgs = []
    k = 0
    with open('X:\\GXB\\20x_and_40x_data\\test.txt') as name_log:
        for line in name_log:
            if k < img_num:
                name = line[0 : line.find('.tif')]
                temp = pool.apply_async(get_item , args = (layers , name , data_path , label_path , dtcwt_fusion_path))
                data.append(temp.get()[0])
                label.append(temp.get()[1])

                if dtcwt_fusion_path != None:
                    dtctw_fusion_imgs.append(temp.get()[2])

            k += 1
    pool.close()
    pool.join()
    if dtcwt_fusion_path == None:
        return data , label
    else:
        return data , label , dtctw_fusion_imgs


def save_result_and_evaluate_2(layers , result_path , weight , data , gen_label , label , img_k):

    if not os.path.exists(result_path + weight + '\\'):
        os.makedirs(result_path + weight + '\\')

    img20x_data = data.cpu().detach().numpy()
    img20x_label = label.cpu().detach().numpy()
    gen20x_label = gen_label.cpu().detach().numpy()
    temp = []
    for k in range(0, len(layers)):
        # temp.append(np.transpose(img20x_data[k * 3: (k + 1) * 3], [1, 2, 0]) * 255)
        temp.append((np.transpose(img20x_data[k * 3 : (k + 1) * 3] , [1 , 2 , 0]) + 1) * 127.5)

    # temp.append(np.transpose(gen20x_label, [1, 2, 0]) * 255)
    temp.append((np.transpose(gen20x_label, [1, 2, 0]) + 1) * 127.5)
    # temp.append(np.transpose(img20x_label, [1, 2, 0]) * 255)
    temp.append((np.transpose(img20x_label, [1, 2, 0]) + 1) * 127.5)

    img20x_data = cv2.hconcat(np.uint8(temp))
    img20x_data = cv2.cvtColor(img20x_data, cv2.COLOR_RGB2BGR)

    layer_0 = int(len(layers) / 2)
    len_layers = len(layers)
    img20x_layer_0 = img20x_data[: , layer_0 * 512 : (layer_0 + 1) * 512 , :]

    gen20x_label = img20x_data[: , len_layers * 512 : (len_layers + 1) * 512 , :]
    img20x_label = img20x_data[:, (len_layers + 1) * 512: (len_layers + 2) * 512, :]


    cv2.imwrite(result_path + weight + '\\' + str(img_k) + '.tif', img20x_data)


    ssim_f = metrics.ssim_m(img20x_label , gen20x_label)
    ssim_0 = metrics.ssim_m(img20x_label , img20x_layer_0)

    cc_f = metrics.correlation_coe(img20x_label , gen20x_label , channel = 3)
    cc_0 = metrics.correlation_coe(img20x_label , img20x_layer_0 , channel = 3)

    return [ssim_0 , ssim_f] , [cc_0 , cc_f]

def save_result_and_evaluate(layers , result_path , weight , data , gen_label , label , img_k):

    if not os.path.exists(result_path + weight + '\\'):
        os.makedirs(result_path + weight + '\\')

    img20x_data = data.cpu().detach().numpy()
    img20x_label = label.cpu().detach().numpy()
    gen20x_label = gen_label.cpu().detach().numpy()
    temp = []
    for k in range(0, len(layers)):
        temp.append(np.transpose(img20x_data[k * 3: (k + 1) * 3], [1, 2, 0]) * 255)

    temp.append(np.transpose(gen20x_label, [1, 2, 0]) * 255)
    temp.append(np.transpose(img20x_label, [1, 2, 0]) * 255)
    img20x_data = cv2.hconcat(np.uint8(temp))
    img20x_data = cv2.cvtColor(img20x_data, cv2.COLOR_RGB2BGR)

    layer_0 = int(len(layers) / 2)
    len_layers = len(layers)
    img20x_layer_0 = img20x_data[: , layer_0 * 512 : (layer_0 + 1) * 512 , :]

    gen20x_label = img20x_data[: , len_layers * 512 : (len_layers + 1) * 512 , :]
    img20x_label = img20x_data[:, (len_layers + 1) * 512: (len_layers + 2) * 512, :]


    cv2.imwrite(result_path + weight + '\\' + str(img_k) + '.tif', img20x_data)

    vif_f = metrics.vifp_mscale(img20x_label , gen20x_label , channel = 3)
    vif_0 = metrics.vifp_mscale(img20x_label , img20x_layer_0 , channel = 3)

    en_g = metrics.entropy(gen20x_label , channel = 3)
    en_f = metrics.entropy(img20x_label , channel = 3)
    en_0 = metrics.entropy(img20x_layer_0 , channel = 3)

    ssim_f = metrics.ssim_m(img20x_label , gen20x_label)
    ssim_0 = metrics.ssim_m(img20x_label , img20x_layer_0)

    cc_f = metrics.correlation_coe(img20x_label , gen20x_label , channel = 3)
    cc_0 = metrics.correlation_coe(img20x_label , img20x_layer_0 , channel = 3)

    sf_g = metrics.spatial_frequency(gen20x_label , channel = 3)
    sf_f = metrics.spatial_frequency(img20x_label , channel = 3)
    sf_0 = metrics.spatial_frequency(img20x_layer_0 , channel = 3)

    sd_g = metrics.standard_deviation(gen20x_label , channel = 3)
    sd_f = metrics.standard_deviation(img20x_label , channel = 3)
    sd_0 = metrics.standard_deviation(img20x_layer_0 , channel = 3)

    return [en_0 , en_g , en_f] , [sd_0 , sd_g , sd_f] , [ssim_0 , ssim_f] , \
           [cc_0 , cc_f] , [sf_0 , sf_g , sf_f] , [vif_0 , vif_f]

def torch_tensor_to_numpy(data):
    data = data.cpu().detach().numpy()
    data = np.transpose(data , [0 , 2 , 3 , 1])
    return data