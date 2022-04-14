import cv2
import os
import multiprocessing.dummy as multi
from multiprocessing import cpu_count
import random
from fusion_metrics.metrics import Metrics

m = Metrics()

result_path = 'X:\\GXB\\paper&patent\\cnn\data\\'
k = 0

def handle_img(path , cz):
    global k
    img = cv2.imread(path + cz , 0)

    origin_img = img.copy()
    size = origin_img.shape

    block_size = 16

    x = random.randint(0 , size[0] - block_size)
    y = random.randint(0 , size[1] - block_size)

    # print(m.standard_deviation(img[x : x + block_size , y : y + block_size]))

    while m.standard_deviation(img[x : x + block_size , y : y + block_size]) < 25:
        x = random.randint(0, size[0] - block_size)
        y = random.randint(0, size[1] - block_size)

    cz = cz[ : cz.find('.tif')]
    cv2.imwrite(result_path + 'c_img_new\\' + cz + '_' + str(0) +'.tif' , img[x : x + block_size , y : y + block_size])
    for i in range(0 , 5):
        img = cv2.GaussianBlur(img, (7 , 7), sigmaX = 2, sigmaY = 2)
        cv2.imwrite(result_path + 'b_img_new\\' + cz + '_' + str(i + 1) + '.tif', img[x: x + block_size, y: y + block_size])

    print(k)
    k += 1


if __name__ == '__main__':
    path = 'X:\\GXB\\20x_and_40x_data\\label\\'
    czs = os.listdir(path)

    pool = multi.Pool(cpu_count())

    for cz in czs:
        pool.apply_async(handle_img , args = (path , cz))
        # cv2.GaussianBlur(blur_region, (h_kernel, w_kernel), sigmaX=sigma_x, sigmaY=sigma_y)


    pool.close()
    pool.join()