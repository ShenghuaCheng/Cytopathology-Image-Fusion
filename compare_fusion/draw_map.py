import numpy as np
import matplotlib.pyplot as plt

def read_file_info(file_path):

    ssim , cc , nmi = [] , [] , []

    with open(file_path) as f:

        for line in f:
            line = line[ : -1]
            line_uints = line.split('\t')
            ssim.append(np.float32(line_uints[1]))
            cc.append(np.float32(line_uints[2]))
            nmi.append(np.float32(line_uints[3]))

    return [ssim , cc , nmi]


def write_test_file():
    path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\gen_0_fusion\\FusionGan_metrics.txt'
    with open(path) as f:

        temp = open('X:\\GXB\\20x_and_40x_data\\test_new.txt' , 'w')

        for line in f:
            line = line[ : -1]
            line_uints = line.split('\t')
            print(line_uints[0])
            temp.write(line_uints[0] + '.tif\n')
        temp.close()


if __name__ == '__main__':
    color = ['red', 'blue', 'green', 'black']

    path = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares_new\\gen_0_fusion\\'

    czs_0 = ['FusionGAN' , 'Pixel2Pixel' , 'SRGAN' , 'our_no_BM' , 'our_BM']
    czs = ['FusionGAN' , 'Pixel2Pixel' , 'SRGAN' , 'our_no_BM' , 'CNN' , 'WT' , 'LPP' , 'DTCWT' , 'our_BM']
    ssims , ccs , nmis = [] , [] , []
    for c in czs_0:

        current_file_name = path + c + '_metrics.txt'
        temp = read_file_info(current_file_name)
        ssims.append(temp[0])
        ccs.append(temp[1])
        nmis.append(temp[2])

    ssims = np.array(ssims)
    ccs = np.array(ccs)
    nmis = np.array(nmis)

    print(np.shape(ssims) , np.shape(ccs) , np.shape(nmis))

    start = 70
    nums = start + 30
    width = 2
    x = [x for x in range(0  , nums - start)]
    plt.plot(x , ssims[0 , start : nums], 'x--', color = 'c' , linewidth=width , label = 'FusionGAN')
    plt.plot(x, ssims[1 , start : nums], 'o--', color = 'b' , linewidth=width , label= 'Pixel2Pixel')
    plt.plot(x, ssims[2 , start : nums], '*--', color ='g' , linewidth=width , label = 'SRGAN')
    # plt.plot(x, nmis[3 , start : nums], 'v--', color='brown' , label= 'unet_patch64')
    # plt.plot(x, nmis[4, start: nums], '.--' , color = 'r', linewidth=width , label='our')
    plt.plot(x, ssims[4, start: nums], 'v--', color = 'r' , label='our')
    # plt.plot(x, ssims[5, start: nums], '+--', color = 'cyan' , label='WT')
    # plt.plot(x, ssims[6, start: nums], '^--', color = 'black' , label='LPP')
    # plt.plot(x, ssims[7, start: nums], 'H--', color = 'dimgray' , label='DTCWT')
    # plt.plot(x, ssims[8, start: nums], '.--' , color = 'r', label='our')
    #
    plt.tick_params(labelsize = 18)
    #
    plt.legend(loc = 'lower right' , prop = {'size' : 18})
    # plt.title('SSIM')
    # plt.legend(bbox_to_anchor=(1.1 , 0), loc = 3, borderaxespad = 0 , prop = {'size' : 24})
    # # plt.legend(loc='SouthWestOutside')
    # plt.xlabel('image')
    # plt.ylabel('Qmi')
    #
    # plt.show()
    # plt.tight_layout()


    print(np.mean(ssims[0 , :]) , np.mean(ssims[1 , :]) , np.mean(ssims[2 , :]) , np.mean(ssims[3 , :]) ,np.mean(ssims[4 , :]))
    print(np.mean(ccs[0, :]), np.mean(ccs[1, :]), np.mean(ccs[2, :]), np.mean(ccs[3, :]) , np.mean(ccs[4, :]))
    print(np.mean(nmis[0, :]), np.mean(nmis[1, :]), np.mean(nmis[2, :]), np.mean(nmis[3, :]) ,np.mean(nmis[4, :]))

    # print(np.mean(ssims[0 , :]) , np.mean(ssims[1 , :]) , np.mean(ssims[2 , :]) , np.mean(ssims[3 , :]) ,
    #       np.mean(ssims[4 , :]) , np.mean(ssims[5 , :]) , np.mean(ssims[6 , :]) , np.mean(ssims[7 , :]) , np.mean(ssims[8 , :]))
    # print(np.mean(ccs[0, :]), np.mean(ccs[1, :]), np.mean(ccs[2, :]), np.mean(ccs[3, :]) ,
    #       np.mean(ccs[4, :]) , np.mean(ccs[5, :]) , np.mean(ccs[6, :]) , np.mean(ccs[7, :]) , np.mean(ccs[8 , :]))
    # print(np.mean(nmis[0, :]), np.mean(nmis[1, :]), np.mean(nmis[2, :]), np.mean(nmis[3, :]) ,
    #       np.mean(nmis[4, :]) , np.mean(nmis[5, :]) , np.mean(nmis[6, :]) , np.mean(nmis[7, :]) , np.mean(nmis[8 , :]))


