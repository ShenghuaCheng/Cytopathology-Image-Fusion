import os
import torch
from model.model import Light
import multiprocessing.dummy as multiprocessing
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from fusion_metrics.metrics import Metrics

run_log_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_3_layers\\'
if not os.path.exists(run_log_path):
    os.makedirs(run_log_path)

metrics = Metrics()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_item(layers , img_name , data_path , label_path):
    img20x_data = []
    for layer in layers:
        temp = cv2.imread(data_path + img_name + '_' + str(layer) + '.tif')
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp = np.transpose(temp, axes=(2, 0, 1)).astype(np.float32) / 255.
        img20x_data.append(temp)
    img_20x_data = np.concatenate(img20x_data, axis=0)

    img_20x_label_path = os.path.join(label_path, img_name + '.tif')
    img_20x_label = cv2.imread(img_20x_label_path)

    # BGR to RGB
    img_20x_label = cv2.cvtColor(img_20x_label, cv2.COLOR_BGR2RGB)
    # H*W*C to C*H*W
    img_20x_label = np.transpose(img_20x_label, axes=(2, 0, 1)).astype(np.float32) / 255.

    return [img_20x_data , img_20x_label]

def read_imgs(layers , img_num = 1000):
    data_path = 'X:\\GXB\\20x_and_40x_data\\split_data\\'
    label_path = 'X:\\GXB\\20x_and_40x_data\\label\\'

    pool = multiprocessing.Pool(cpu_count())
    data , label = [] , []
    k = 0
    with open('X:\\GXB\\20x_and_40x_data\\test.txt') as name_log:
        for line in name_log:
            if k < img_num:
                name = line[0 : line.find('.tif')]
                temp = pool.apply_async(get_item , args = (layers , name , data_path , label_path))
                data.append(temp.get()[0])
                label.append(temp.get()[1])
            k += 1
    pool.close()
    pool.join()
    return data , label

def test_model_mean():
    layers = [-2, 0, 2]
    datas, labels = read_imgs(layers, img_num = 50)
    print(np.shape(datas), np.shape(labels))

    model_path = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_3_layers\\'
    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_3_layers_multi_models\\'
    test_list = ['netG_epoch_4_3710.pth', 'netG_epoch_4_3703.pth', 'netG_epoch_4_3696.pth', 'netG_epoch_4_3690.pth',
                 'netG_epoch_4_3684.pth']

    log_writer = SummaryWriter(result_path + 'fusion_3_layers_multi_models.log')
    log_file = open(result_path + 'fusion_3_layers_multi_models.txt' , 'w')

    model_g = Light(9, 3)

    for index in range(0 , len(test_list)):
        current_weight_path = model_path + test_list[index]
        model_g.load_state_dict(torch.load(current_weight_path))
        model_g.cuda()
        en_avg = np.array([0. , 0 , 0])
        sd_avg = np.array([0. , 0 , 0])
        ssim_avg = np.array([0., 0 ])
        cc_avg = np.array([0. , 0])
        sf_avg = np.array([0. , 0 , 0])
        vif_avg = np.array([0. , 0 ])
        k = 0
        for data, label in zip(datas, labels):
            data = [data]
            label = [label]
            data = torch.from_numpy(np.array(data))
            label = torch.from_numpy(np.array(label))
            print(np.shape(data))
            data = data.cuda()
            with torch.no_grad():
                gen_label = model_g(data)
                en, sd, ssim, cc, sf, vif = save_result_and_evaluate(layers, result_path, test_list[index], data[0], gen_label[0],
                                                                     label[0], k)
                en_avg += np.array(en)
                sd_avg += np.array(sd)
                ssim_avg += np.array(ssim)
                cc_avg += np.array(cc)
                sf_avg += np.array(sf)
                vif_avg += np.array(vif)
            k += 1
        en_avg /= k
        sd_avg /= k
        ssim_avg /= k
        cc_avg /= k
        sf_avg /= k
        vif_avg /= k

        log_writer.add_scalars('scalar/en', {'en_0': en_avg[0], 'en_g': en_avg[1], 'en_f': en_avg[2]}, index)
        log_file.write(
            str(index) + '-' + 'en_0:' + str(en_avg[0]) + 'en_g:' + str(en_avg[1]) + 'en_f:' + str(en_avg[2]) + '\n')
        log_writer.add_scalars('scalar/sd', {'sd_0': sd_avg[0], 'sd_g': sd_avg[1], 'sd_f': sd_avg[2]}, index)
        log_file.write(
            str(index) + '-' + 'sd_0:' + str(sd_avg[0]) + 'sd_g:' + str(sd_avg[1]) + 'sd_f:' + str(sd_avg[2]) + '\n')
        log_writer.add_scalars('scalar/ssim', {'ssim_0': ssim_avg[0], 'ssim_f': ssim_avg[1]}, index)
        log_file.write(str(index) + '-' + 'ssim_0:' + str(ssim_avg[0]) + 'ssim_f:' + str(ssim_avg[1]) + '\n')
        log_writer.add_scalars('scalar/cc', {'cc_0': cc_avg[0], 'cc_f': cc_avg[1]}, index)
        log_file.write(str(index) + '-' + 'cc_0:' + str(cc_avg[0]) + 'cc_f:' + str(cc_avg[1]) + '\n')
        log_writer.add_scalars('scalar/sf', {'sf_0': sf_avg[0], 'sf_g': sf_avg[1], 'sf_f': sf_avg[2]}, index)
        log_file.write(
            str(index) + '-' + 'sf_0:' + str(sf_avg[0]) + 'sf_g:' + str(sf_avg[1]) + 'sf_f:' + str(sf_avg[2]) + '\n')
        log_writer.add_scalars('scalar/vif', {'vif_0': vif_avg[0], 'vif_f': vif_avg[1]}, index)
        log_file.write(str(index) + '-' + 'vif_0:' + str(vif_avg[0]) + 'vif_f:' + str(vif_avg[1]) + '\n')

    log_writer.close()
    log_file.close()

def test_model():
    layers = [-2 , 0 , 2]
    datas , labels = read_imgs(layers , img_num = 10)
    print(np.shape(datas) , np.shape(labels))

    model_path = 'X:\\GXB\\20x_and_40x_data\\checkpoints\\fusion_3_layers\\'
    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_3_layers_multi_imgs\\'
    test_list = ['netG_epoch_4_3710.pth']
                 #'netG_epoch_2_2159.pth' , 'netG_epoch_2_2160.pth' , 'netG_epoch_2_2161.pth' , 'netG_epoch_2_2162.pth']

    log_writer = SummaryWriter(result_path + 'fusion_3_layers_multi_imgs.log')
    log_file = open(result_path + 'fusion_3_layers_multi_imgs.txt' , 'w')

    model_g = Light(9 , 3)

    for weight in test_list:
        current_weight_path = model_path + weight
        model_g.load_state_dict(torch.load(current_weight_path))
        model_g.cuda()
        k = 0
        for data , label in zip(datas , labels):
            data = [data]
            label = [label]
            data = torch.from_numpy(np.array(data))
            label = torch.from_numpy(np.array(label))
            print(np.shape(data))
            data = data.cuda()
            with torch.no_grad():
                gen_label = model_g(data)
                en_avg, sd_avg, ssim_avg, cc_avg, sf_avg, vif_avg = save_result_and_evaluate(layers , result_path , weight , data[0] , gen_label[0] , label[0] , k)

            log_writer.add_scalars('scalar/en', {'en_0': en_avg[0], 'en_g': en_avg[1], 'en_f': en_avg[2]}, k)
            log_file.write(str(k) + '-' + 'en_0:' + str(en_avg[0]) + 'en_g:' + str(en_avg[1]) + 'en_f:' + str(en_avg[2]) + '\n')
            log_writer.add_scalars('scalar/sd', {'sd_0': sd_avg[0], 'sd_g': sd_avg[1], 'sd_f': sd_avg[2]}, k)
            log_file.write(str(k) + '-' + 'sd_0:' + str(sd_avg[0]) + 'sd_g:' + str(sd_avg[1]) + 'sd_f:' + str(sd_avg[2]) +'\n' )
            log_writer.add_scalars('scalar/ssim', {'ssim_0': ssim_avg[0], 'ssim_f': ssim_avg[1]}, k)
            log_file.write(str(k) + '-' + 'ssim_0:' + str(ssim_avg[0]) + 'ssim_f:' + str(ssim_avg[1]) + '\n')
            log_writer.add_scalars('scalar/cc', {'cc_0': cc_avg[0], 'cc_f': cc_avg[1]}, k)
            log_file.write(str(k) + '-' + 'cc_0:' + str(cc_avg[0]) + 'cc_f:' + str(cc_avg[1]) + '\n')
            log_writer.add_scalars('scalar/sf', {'sf_0': sf_avg[0], 'sf_g': sf_avg[1], 'sf_f': sf_avg[2]}, k)
            log_file.write(str(k) + '-' + 'sf_0:' + str(sf_avg[0]) + 'sf_g:' + str(sf_avg[1]) + 'sf_f:' + str(sf_avg[2]) + '\n')
            log_writer.add_scalars('scalar/vif', {'vif_0': vif_avg[0], 'vif_f': vif_avg[1]}, k)
            log_file.write(str(k) + '-' + 'vif_0:' + str(vif_avg[0]) + 'vif_f:' + str(vif_avg[1]) + '\n')
            k += 1

    log_writer.close()
    log_file.close()

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

    return [en_0 , en_g , en_f] , [sd_0 , sd_g , sd_f] , [ssim_0 , ssim_f] , [cc_0 , cc_f] , [sf_0 , sf_g , sf_f] , [vif_0 , vif_f]

if __name__ == '__main__':
    test_model_mean()
