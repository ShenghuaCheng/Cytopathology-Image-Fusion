import openslide
import numpy as np
from multiprocessing import dummy
from multiprocessing import cpu_count
import cv2
import sys

result_path = r'X:\GXB\20x_and_40x_data\R'


class WholeSlideReader():

    def __init__(self, path, cz, items_per_it=20, block_size=512, redu_ratio=1. / 4, level=0):

        self.path = path
        self.cz = cz
        self.items_per_it = items_per_it
        self.block_size = block_size
        self.redu_ratio = redu_ratio
        self.redu_size = int(block_size * redu_ratio)
        self.level = level

        print(path + cz)
        self.ors = openslide.OpenSlide(path + cz)
        self.slide_size = self.ors.level_dimensions[level]

        self.w_nums = (self.slide_size[0] - block_size) // (self.block_size - self.redu_size) + 1
        self.h_nums = (self.slide_size[1] - block_size) // (self.block_size - self.redu_size) + 1
        self.get_position_list()
        self.index = 0
        self.read_flag = True
        self.predicted_imgs = []

    def get_position_list(self):
        self.position_list = []
        for w in range(0, self.w_nums):
            for h in range(0, self.h_nums):
                self.position_list.append(
                    (w * (self.block_size - self.redu_size), h * (self.block_size - self.redu_size)))

    def read_item(self, index):
        img = self.ors.read_region(self.position_list[index], self.level, (self.block_size, self.block_size))
        img = np.array(img)
        img = img[:, :, 0 : 3]
        img = img[:, :, ::-1]
        img = np.float32(img) / 255.
        return img

    def read_slide(self):
        i = 0
        imgs = []
        pool = dummy.Pool(cpu_count())
        while i < self.items_per_it and self.index < len(self.position_list):
            imgs.append(pool.apply_async(self.read_item, args=(self.index,)).get())
            self.index += 1
            i += 1
        pool.close()
        pool.join()
        if self.index >= len(self.position_list):
            self.read_flag = False
        sys.stdout.flush()
        print('split img %d/%d \r' % (self.index , len(self.position_list)))
        return np.array(imgs)

    def put_predicted_imgs(self, predicted_imgs):
        i = 0
        for img in predicted_imgs:
            self.predicted_imgs.append(img)
            i += 1

    def generate_whole_slide(self):

        fusion_map = np.zeros(self.slide_size + (3 , ) , dtype = np.uint8)
        for position , img in zip(self.position_list , self.predicted_imgs):

            img = np.uint8(img * 255)
            img = img[ : , : , ::-1]

            #处理 w 方向上的重叠区域
            if position[0] != 0:
                left_region = fusion_map[position[1] : position[1] + self.block_size , position[0] : position[0] + self.redu_size]
                right_region = img[: , 0 : self.redu_size]
                f_region = np.zeros(right_region.shape , dtype = np.uint8)
                for w in range(self.redu_size):
                    for h in range(0 , self.block_size):
                        f_region[h , w] = left_region[h , w] * ((self.redu_size - w) / self.redu_size) + \
                                          right_region[h , w] * (w / self.redu_size)
                fusion_map[position[1]: position[1] + self.block_size,
                position[0] : position[0] + self.redu_size] = f_region
            else:
                fusion_map[position[1]: position[1] + self.block_size,
                position[0] : position[0] + self.redu_size] = img[: , 0 : self.redu_size]

            #处理 h 方向上的重叠区域
            if position[1] != 0:
                up_region = fusion_map[position[1] : position[1] + self.redu_size , position[0] : position[0] + self.block_size]
                down_region = img[0 : self.redu_size , :]
                f_region = np.zeros(up_region.shape , dtype = np.uint8)
                for h in range(self.redu_size):
                    for w in range(0 , 512):
                        f_region[h , w] = up_region[h , w] * ((self.redu_size - h) / self.redu_size) + \
                                          down_region[h , w] * (h / self.redu_size)
                fusion_map[position[1]: position[1] + self.redu_size,
                position[0]: position[0] + self.block_size] = f_region
            else:
                fusion_map[position[1]: position[1] + self.redu_size ,
                position[0] : position[0] + self.block_size] = img[0 : self.redu_size , :]

            fusion_map[position[1] + self.redu_size : position[1] + self.block_size,
            position[0] + self.redu_size : position[0] + self.block_size] = img[self.redu_size : , self.redu_size : ]
        fusion_map = cv2.resize(fusion_map , None , fx = 0.1 , fy = 0.1)

        cv2.imwrite(result_path + '\\fusion_map.tif' , fusion_map)


if __name__ == '__main__':
    path = r'E:\20x_and_40x_new\20x_tiff'
    cz = '10140015'
    # ws = WholeSlideReader(path + '\\' + cz + '\\' , cz + '_Wholeslide_0.tif' , items_per_it = 100)
    #
    # while ws.read_flag:
    #     imgs = ws.read_slide()
    #     predicted_imgs = imgs.copy()
    #     ws.put_predicted_imgs(predicted_imgs)
    # ws.generate_whole_slide()

    img_names = r'E:\20x_and_40x_new\20x_tiff\10140015\10140015_Wholeslide_0.tif'
    ors = openslide.OpenSlide(img_names)

    ratio = 5

    width , height = ors.level_dimensions[0]
    width , height = width // 5 , height // 5
    print('read  ... ')
    imgs = []
    for i in range(0 , 5):
        temp = []
        for j in range(0 , 5):
            img = cv2.imread(result_path + '\\temp_' + str(i) + '_' + str(j) + '.tif')
            temp.append(img)
        imgs.append(cv2.vconcat(temp))

    final_img = cv2.hconcat(imgs)
    cv2.imwrite(result_path + '\\final_img.tif' , final_img)


