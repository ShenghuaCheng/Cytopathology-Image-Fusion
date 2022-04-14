from tensorboardX import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt


def get_metrics_info(path):
    en , sd , ssim , cc , sf , vif = [] , [] , [] , [] , [] , []
    with open(path) as log_file:
        for line in log_file:
            line = line[:-1]
            print(line)
            if line.find('en') != -1:
                en_0 = line[line.find('en_0:') + 5 : line.find('en_g')]
                en_g = line[line.find('en_g:') + 5 : line.find('en_f')]
                en_f = line[line.find('en_f:') + 5 : line.find('en_d')]
                en_d = line[line.find('en_d:') + 5 :]
                print(line[0] , en_0 , en_g , en_f , en_d)
                en.append([float(en_0) , float(en_g) , float(en_f) , float(en_d)])
            if line.find('sd') != -1:
                sd_0 = line[line.find('sd_0:') + 5 : line.find('sd_g')]
                sd_g = line[line.find('sd_g:') + 5 : line.find('sd_f')]
                sd_f = line[line.find('sd_f:') + 5 : line.find('sd_d')]
                sd_d = line[line.find('sd_d:') + 5 : ]
                print(line[0] , sd_0 , sd_g , sd_f , sd_d)
                sd.append([float(sd_0) , float(sd_g) , float(sd_f) , float(sd_d)])
            if line.find('ssim') != -1:
                ssim_0 = line[line.find('ssim_0:') + 7 : line.find('ssim_f:')]
                ssim_f = line[line.find('ssim_f:') + 7 : line.find('ssim_d:')]
                ssim_d = line[line.find('ssim_d:') + 7 : ]
                print(line[0] , ssim_0 , ssim_f , ssim_d)
                ssim.append([float(ssim_0) , float(ssim_f) , float(ssim_d)])
            if line.find('cc') != -1:
                cc_0 = line[line.find('cc_0:') + 5 : line.find('cc_f:')]
                cc_f = line[line.find('cc_f:') + 5 : line.find('cc_d:')]
                cc_d = line[line.find('cc_d:') + 5 : ]
                print(line[0] , cc_0 , cc_f , cc_d)
                cc.append([float(cc_0) , float(cc_f) , float(cc_d)])
            if line.find('sf') != -1:
                sf_0 = line[line.find('sf_0:') + 5 : line.find('sf_g:')]
                sf_g = line[line.find('sf_g:') + 5: line.find('sf_f:')]
                sf_f = line[line.find('sf_f:') + 5: line.find('sf_d')]
                sf_d = line[line.find('sf_d') + 4 : ]
                print(line[0] , float(sf_0) , float(sf_g) , float(sf_f) , sf_d)
                sf.append([float(sf_0), float(sf_g) , float(sf_f) , float(sf_d)])
            if line.find('vif') != -1:
                vif_0 = line[line.find('vif_0:') + 6 : line.find('vif_f:')]
                vif_f = line[line.find('vif_f:') + 6 : line.find('vif_d')]
                vif_d = line[line.find('vif_d') + 5 : ]
                print(line[0] , vif_0 , vif_f , vif_d)
                vif.append([float(vif_0) , float(vif_f) , float(vif_d)])
    return np.array(en) , np.array(sd) , np.array(ssim) , np.array(cc) , np.array(sf) , np.array(vif)

def generate_multi_models_map():
    log_files = ['fusion_0_layer_multi_models', 'fusion_0_layer_patch_multi_models', 'fusion_-2_2_15sr_layer_multi_models',
                 'fusion_3_layers_multi_models', 'fusion_11_layers_secondtime_multi_models']

    log_tag = ['fu_0', 'fu_0_patch', 'fu_-2_2', 'fu_3_layers', 'fu_11_layers']

    path = 'X:\\GXB\\20x_and_40x_data\\test_result\\'

    writer = SummaryWriter(path + 'multi_models_log')

    for i in range(0 , len(log_files)):
        log_file , tag = log_files[i] , log_tag[i]
        print(log_files , tag)
        l = os.listdir(path + log_file + '\\')
        l = [x for x in l if x.find('.txt') != -1]
        en, sd, ssim, cc, sf, vif = get_metrics_info(path + log_file + '\\' + l[0])

        k = 0
        for (en_k , sd_k , ssim_k , cc_k , sf_k , vif_k) in zip(en , sd , ssim , cc , sf , vif):
            writer.add_scalars('scalar/en', {tag + '_en_g': en_k[1]}, k)
            writer.add_scalars('scalar/sd', {tag + '_sd_g': sd_k[1]}, k)
            writer.add_scalars('scalar/ssim', {tag + '_ssim_f': ssim_k[1]} , k)
            writer.add_scalars('scalar/cc', {tag + '_cc_f': cc_k[1]}, k)
            writer.add_scalars('scalar/sf', {tag + '_sf_g': sf_k[1]}, k)
            writer.add_scalars('scalar/vif', {tag + '_vif_f': vif_k[1]}, k)

            if i == 0:
                writer.add_scalars('scalar/en', {'en_0': en_k[0] , 'en_f' : en_k[2]}, k)
                writer.add_scalars('scalar/sd', {'sd_0': sd_k[0] , 'sd_f' : sd_k[2]}, k)
                writer.add_scalars('scalar/ssim', {'ssim_0': ssim_k[0]}, k)
                writer.add_scalars('scalar/cc', {'cc_0': cc_k[0]}, k)
                writer.add_scalars('scalar/sf', {'sf_0': sf_k[0] , 'sf_f' : sf_k[2]}, k)
                writer.add_scalars('scalar/vif', {'vif_0': vif_k[0]}, k)

            k+= 1

    writer.close()

def generate_multi_imgs_map():

    log_files = ['fusion_0_layer_multi_imgs', 'fusion_0_layer_patch_multi_imgs', 'fusion_-2_2_15sr_layer_multi_imgs',
                 'fusion_3_layers_multi_imgs', 'fusion_11_layers_secondtime_multi_imgs']

    log_tag = ['fu_0', 'fu_0_patch', 'fu_-2_2', 'fu_3_layers', 'fu_11_layers']

    path = 'X:\\GXB\\20x_and_40x_data\\test_result\\'

    writer = SummaryWriter(path + 'multi_imgs_log')

    for i in range(0 , len(log_files)):
        log_file , tag = log_files[i] , log_tag[i]
        print(log_files , tag)
        l = os.listdir(path + log_file + '\\')
        l = [x for x in l if x.find('.txt') != -1]
        en, sd, ssim, cc, sf, vif = get_metrics_info(path + log_file + '\\' + l[0])

        k = 0
        for (en_k , sd_k , ssim_k , cc_k , sf_k , vif_k) in zip(en , sd , ssim , cc , sf , vif):
            writer.add_scalars('scalar/en', {tag + '_en_g': en_k[1]}, k)
            writer.add_scalars('scalar/sd', {tag + '_sd_g': sd_k[1]}, k)
            writer.add_scalars('scalar/ssim', {tag + '_ssim_f': ssim_k[1]} , k)
            writer.add_scalars('scalar/cc', {tag + '_cc_f': cc_k[1]}, k)
            writer.add_scalars('scalar/sf', {tag + '_sf_g': sf_k[1]}, k)
            writer.add_scalars('scalar/vif', {tag + '_vif_f': vif_k[1]}, k)

            if i == 0:
                writer.add_scalars('scalar/en', {'en_0': en_k[0] , 'en_f' : en_k[2]}, k)
                writer.add_scalars('scalar/sd', {'sd_0': sd_k[0] , 'sd_f' : sd_k[2]}, k)
                writer.add_scalars('scalar/ssim', {'ssim_0': ssim_k[0]}, k)
                writer.add_scalars('scalar/cc', {'cc_0': cc_k[0]}, k)
                writer.add_scalars('scalar/sf', {'sf_0': sf_k[0] , 'sf_f' : sf_k[2]}, k)
                writer.add_scalars('scalar/vif', {'vif_0': vif_k[0]}, k)

            k+= 1

    writer.close()


def draw_multi_imgs_map():

    log_tag = ['fu_0', 'fu_0_patch', 'fu_-2_2', 'fu_3_layers', 'fu_11_layers']
    color = ['red' , 'blue' , 'green' , 'black' , 'yellow']

    path = 'X:\\GXB\\20x_and_40x_data\\test_result\\'

    log_file= 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_-3-3_layers_imgs\\'

    l = os.listdir(log_file)
    l = [x for x in l if x.find('.txt') != -1]
    en, sd, ssim, cc, sf, vif = get_metrics_info(log_file + l[0])
    x = [t for t in range(0 , 10)]

    plt.plot(x , vif[ : , 1], c=color[0], label = 'fu_-3-3')
    plt.plot(x, vif[:, 0], c=color[1], label = log_tag[0] + '_0')
    plt.plot(x , vif[: , 2] , c = color[2] , label = 'fu_dtcwt')
    if False:
        plt.plot(x, vif[:, 2], c=color[3], label=log_tag[4] + '_f')
        # plt.plot(x, ssim[:, 0], c='magenta', label=tag)

    plt.legend(loc = 'best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("multi_imgs_vif")

    plt.show()

def draw_multi_models_map():
    log_tag = ['fu_0', 'fu_0_patch', 'fu_-2_2', 'fu_3_layers', 'fu_11_layers']
    color = ['red' , 'blue' , 'green' , 'black' , 'yellow']

    path = 'X:\\GXB\\20x_and_40x_data\\test_result\\'

    log_file= 'X:\\GXB\\20x_and_40x_data\\test_result\\fusion_-3-3_layers_models\\'

    l = os.listdir(log_file)
    l = [x for x in l if x.find('.txt') != -1]
    en, sd, ssim, cc, sf, vif = get_metrics_info(log_file + l[0])
    x = [t for t in range(0 , 5)]

    plt.plot(x , sf[ : , 1], c=color[0], label = 'fu_-3-3')
    plt.plot(x, sf[:, 0], c=color[1], label = log_tag[0] + '_0')
    plt.plot(x , sf[: , 3] , c = color[2] , label = 'fu_dtcwt')
    if True:
        plt.plot(x, sf[:, 2], c=color[3], label=log_tag[4] + '_f')
        # plt.plot(x, ssim[:, 0], c='magenta', label=tag)

    plt.legend(loc = 'best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("multi_models_sf")

    plt.show()

if __name__ == '__main__':
    # generate_multi_imgs_map()
    # generate_multi_models_map()
    # draw_multi_imgs_map()
    # draw_multi_imgs_map()
    draw_multi_models_map()