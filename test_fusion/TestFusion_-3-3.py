import os
import torch
from model.model import Light
from train_and_test.public_code import *

metrics = Metrics()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_models():
    layers = [-3 , -2 , -1, 0, 1 , 2 , 3]
    dtcwt_fusion_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\multi_img_fusion_test_img\\dtcwt_fusion_30_imgs\\-3-3\\'
    datas, labels , dtcwt_fusion_imgs = read_imgs(layers, img_num = 30 , dtcwt_fusion_path = dtcwt_fusion_path)
    print(np.shape(datas), np.shape(labels))

    model_path = 'V:\\fusion_-3-3_unet\\checkpoints\\'
    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_-3-3_layers_models\\'
    test_list = ['netG_epoch_8_5286.pth','netG_epoch_8_5270.pth','netG_epoch_8_5268.pth','netG_epoch_8_5266.pth','netG_epoch_8_5265.pth']

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    log_file = open(result_path + 'fusion_-3-3_layers_models.txt' , 'w')

    model_g = Light(21, 3)

    for index in range(0 , len(test_list)):
        current_weight_path = model_path + test_list[index]
        model_g.load_state_dict(torch.load(current_weight_path))
        model_g.cuda()
        en_avg = np.array([0. , 0 , 0 , 0])
        sd_avg = np.array([0. , 0 , 0 , 0])
        ssim_avg = np.array([0., 0 , 0])
        cc_avg = np.array([0. , 0 , 0])
        sf_avg = np.array([0. , 0 , 0 , 0])
        vif_avg = np.array([0. , 0 , 0])
        k = 0
        for data, label , dtcwt_fusion_img in zip(datas, labels , dtcwt_fusion_imgs):
            data = [data]
            label = [label]
            data = torch.from_numpy(np.array(data))
            label = torch.from_numpy(np.array(label))
            print(np.shape(data))
            data = data.cuda()
            with torch.no_grad():
                gen_label = model_g(data)
                en, sd, ssim, cc, sf, vif = save_result_and_evaluate(layers, result_path, test_list[index], data[0], gen_label[0],
                                                                     label[0], dtcwt_fusion_img ,  k)
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

        log_file.write(
            str(index) + '-' + 'en_0:' + str(en_avg[0]) + 'en_g:' + str(en_avg[1]) + 'en_f:' + str(en_avg[2]) + 'en_d:' + str(en_avg[3]) + '\n')
        log_file.write(
            str(index) + '-' + 'sd_0:' + str(sd_avg[0]) + 'sd_g:' + str(sd_avg[1]) + 'sd_f:' + str(sd_avg[2]) + 'sd_d:' + str(sd_avg[3]) + '\n')
        log_file.write(str(index) + '-' + 'ssim_0:' + str(ssim_avg[0]) + 'ssim_f:' + str(ssim_avg[1]) + 'ssim_d:' + str(ssim_avg[2]) + '\n')
        log_file.write(str(index) + '-' + 'cc_0:' + str(cc_avg[0]) + 'cc_f:' + str(cc_avg[1]) + 'cc_d:' + str(cc_avg[2]) + '\n')
        log_file.write(
            str(index) + '-' + 'sf_0:' + str(sf_avg[0]) + 'sf_g:' + str(sf_avg[1]) + 'sf_f:' + str(sf_avg[2]) + 'sf_d' + str(sf_avg[3]) + '\n')
        log_file.write(str(index) + '-' + 'vif_0:' + str(vif_avg[0]) + 'vif_f:' + str(vif_avg[1]) + 'vif_d' + str(vif_avg[2]) + '\n')

    log_file.close()

def test_imgs():
    layers = [-3 , -2 , -1, 0, 1 , 2 , 3]
    dtcwt_fusion_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\multi_img_fusion_test_img\\dtcwt_fusion_30_imgs\\-3-3\\'
    datas, labels , dtcwt_fusion_imgs = read_imgs(layers, img_num = 10 , dtcwt_fusion_path = dtcwt_fusion_path)
    print(np.shape(datas), np.shape(labels))

    model_path = 'V:\\fusion_-3-3_unet\\checkpoints\\'
    result_path = 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_-3-3_layers_imgs\\'
    test_list = ['netG_epoch_8_5286.pth']
                 #'netG_epoch_2_2159.pth' , 'netG_epoch_2_2160.pth' , 'netG_epoch_2_2161.pth' , 'netG_epoch_2_2162.pth']

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    log_file = open(result_path + 'fusion_-3-3_layers_imgs.txt' , 'w')

    model_g = Light(21 , 3)

    for weight in test_list:
        current_weight_path = model_path + weight
        model_g.load_state_dict(torch.load(current_weight_path))
        model_g.cuda()
        k = 0
        for data , label , dtcwt_fusion_img in zip(datas , labels , dtcwt_fusion_imgs):
            data = [data]
            label = [label]
            data = torch.from_numpy(np.array(data))
            label = torch.from_numpy(np.array(label))
            print(np.shape(data))
            data = data.cuda()
            with torch.no_grad():
                gen_label = model_g(data)
                en_avg, sd_avg, ssim_avg, cc_avg, sf_avg, vif_avg = save_result_and_evaluate(layers , result_path , weight , data[0] , gen_label[0] , label[0] ,dtcwt_fusion_img, k)

            log_file.write(
                str(k) + '-' + 'en_0:' + str(en_avg[0]) + 'en_g:' + str(en_avg[1]) + 'en_f:' + str(
                    en_avg[2]) + 'en_d:' + str(en_avg[3]) + '\n')
            log_file.write(
                str(k) + '-' + 'sd_0:' + str(sd_avg[0]) + 'sd_g:' + str(sd_avg[1]) + 'sd_f:' + str(
                    sd_avg[2]) + 'sd_d:' + str(sd_avg[3]) + '\n')
            log_file.write(
                str(k) + '-' + 'ssim_0:' + str(ssim_avg[0]) + 'ssim_f:' + str(ssim_avg[1]) + 'ssim_d:' + str(
                    ssim_avg[2]) + '\n')
            log_file.write(str(k) + '-' + 'cc_0:' + str(cc_avg[0]) + 'cc_f:' + str(cc_avg[1]) + 'cc_d:' + str(
                cc_avg[2]) + '\n')
            log_file.write(
                str(k) + '-' + 'sf_0:' + str(sf_avg[0]) + 'sf_g:' + str(sf_avg[1]) + 'sf_f:' + str(
                    sf_avg[2]) + 'sf_d' + str(sf_avg[3]) + '\n')
            log_file.write(str(k) + '-' + 'vif_0:' + str(vif_avg[0]) + 'vif_f:' + str(vif_avg[1]) + 'vif_d' + str(
                vif_avg[2]) + '\n')

            k += 1

    log_file.close()

def save_result_and_evaluate(layers , result_path , weight , data , gen_label , label , dtcwt_fusion_img , img_k):

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

    img20x_data = cv2.hconcat((img20x_data , dtcwt_fusion_img))
    cv2.imwrite(result_path + weight + '\\' + str(img_k) + '.tif', img20x_data)

    vif_f = metrics.vifp_mscale(img20x_label , gen20x_label , channel = 3)
    vif_0 = metrics.vifp_mscale(img20x_label , img20x_layer_0 , channel = 3)
    vif_dtcwt = metrics.vifp_mscale(img20x_label , dtcwt_fusion_img , channel = 3)

    en_g = metrics.entropy(gen20x_label , channel = 3)
    en_f = metrics.entropy(img20x_label , channel = 3)
    en_0 = metrics.entropy(img20x_layer_0 , channel = 3)
    en_dtcwt = metrics.entropy(dtcwt_fusion_img , channel = 3)

    ssim_f = metrics.ssim_m(img20x_label , gen20x_label)
    ssim_0 = metrics.ssim_m(img20x_label , img20x_layer_0)
    ssim_dtcwt = metrics.ssim_m(img20x_label , dtcwt_fusion_img)

    cc_f = metrics.correlation_coe(img20x_label , gen20x_label , channel = 3)
    cc_0 = metrics.correlation_coe(img20x_label , img20x_layer_0 , channel = 3)
    cc_dtcwt = metrics.correlation_coe(img20x_label , dtcwt_fusion_img , channel = 3)

    sf_g = metrics.spatial_frequency(gen20x_label , channel = 3)
    sf_f = metrics.spatial_frequency(img20x_label , channel = 3)
    sf_0 = metrics.spatial_frequency(img20x_layer_0 , channel = 3)
    sf_dtcwt = metrics.spatial_frequency(dtcwt_fusion_img , channel = 3)

    sd_g = metrics.standard_deviation(gen20x_label , channel = 3)
    sd_f = metrics.standard_deviation(img20x_label , channel = 3)
    sd_0 = metrics.standard_deviation(img20x_layer_0 , channel = 3)
    sd_dtcwt = metrics.standard_deviation(dtcwt_fusion_img , channel = 3)

    return [en_0 , en_g , en_f , en_dtcwt] , [sd_0 , sd_g , sd_f , sd_dtcwt] , [ssim_0 , ssim_f , ssim_dtcwt] \
        , [cc_0 , cc_f , cc_dtcwt] , [sf_0 , sf_g , sf_f , sf_dtcwt] , [vif_0 , vif_f , vif_dtcwt]

if __name__ == '__main__':
    test_models()
    test_imgs()