import numpy as np
import cv2

class LaplacianBlending():

    def __init__(self , left_img , right_img , blend_mask , _levels):
        self.left_img = left_img
        self.right_img = right_img
        self.blend_mask = blend_mask
        self._levels = _levels

        self.build_pyramids()
        self.blend_lap_pyrs()

    def reconstruct_img_from_lappyramid(self): #通过拉普拉斯金字塔进行图像的重建
        current_img = self.result_highest_level
        for l in range(self._levels - 1 , -1 , -1):
            up = cv2.pyrUp(current_img , dstsize = self.result_lap_pyr[l].shape[:2])
            current_img = up + self.result_lap_pyr[l]
        return current_img

    def blend(self):
        return self.reconstruct_img_from_lappyramid()

    def caculate_region_gradient(self , region):

        gradient = 0
        rows , cols = np.shape(region)
        for r in range(1 , rows):
            for c in range(1 , cols):
                gradient += np.sqrt((pow(region[r , c] - region[r - 1 , c] , 2)  + pow(region[r , c] - region[r , c - 1] , 2)) / 2)
        return gradient

    def caculate_region_power(self , region):

        w = 1 / 16 * np.array([1 , 2 , 1 , 2 , 4 , 2 , 1 , 2 , 1])
        region = region.reshape((region.shape[0] * region.shape[1] , ))
        region = np.abs(region) * w
        return np.sum(region)

    def fusion_highest_layer(self , left_highest_level , right_highest_level , m = 3 , n = 3):

        assert m % 2 == 1 and n % 2 == 1 #m与n必须为奇数

        fusion_highest_layer = np.zeros(np.shape(left_highest_level) , dtype = np.float)

        print(np.shape(left_highest_level) , np.shape(right_highest_level))
        #将多通道转化为单通道进行处理
        gray_left_highest_level = np.sum(left_highest_level , axis = 2) / 3
        gray_right_highest_level = np.sum(right_highest_level , axis = 2) / 3

        rows, cols = np.shape(gray_left_highest_level)

        assert m <= rows and n <= cols

        for r in range(0 , rows):
            for c in range(0 , cols):
                #对于每个点，求取m * n区域的平均梯度
                if r - int(m / 2) < 0:
                    r_s = 0
                    r_e = r_s + m
                elif r + int(m / 2) >= rows:
                    r_e = rows
                    r_s = rows - m
                else:
                    r_s = r - int(m / 2)
                    r_e = r + int(m / 2)

                if c - int(n / 2) < 0:
                    c_s = 0
                    c_e = c_s + n
                elif c + int(n / 2) >= cols:
                    c_e = cols
                    c_s = cols - n
                else:
                    c_s = c - int(n / 2)
                    c_e = c + int(n / 2)

                region_left = gray_left_highest_level[r_s : r_e , c_s : c_e]
                region_right = gray_right_highest_level[r_s : r_e , c_s : c_e]

                gradient_left = self.caculate_region_gradient(region_left)
                gradient_right = self.caculate_region_gradient(region_right)

                if gradient_left > gradient_right:
                    fusion_highest_layer[r , c , :] = left_highest_level[r , c , :]
                else:
                    fusion_highest_layer[r, c, :] = right_highest_level [r, c, :]
        return fusion_highest_layer


    def fusion_l_layer(self , left_img , right_img , p = 1 , q = 1):
        print(np.shape(left_img), np.shape(right_img))

        fusion_l_layer = np.zeros(np.shape(left_img), dtype=np.float)

        print(np.shape(left_img), np.shape(right_img))
        # 将多通道转化为单通道进行处理
        gray_left_img = np.sum(left_img, axis=2) / 3
        gray_right_img = np.sum(right_img, axis=2) / 3

        rows, cols = np.shape(gray_left_img)

        for r in range(0, rows):
            for c in range(0, cols):
                # 对于每个点，求取m * n区域的平均梯度
                if r - p < 0:
                    r_s = 0
                    r_e = r + 2 * p + 1
                elif r + p >= rows:
                    r_e = rows
                    r_s = rows - 2 * p - 1
                else:
                    r_s = r - p
                    r_e = r + p + 1

                if c - q < 0:
                    c_s = 0
                    c_e = c_s + 2 * q + 1
                elif c + q >= cols:
                    c_e = cols
                    c_s = cols - 2  * q - 1
                else:
                    c_s = c - q
                    c_e = c + q + 1

                region_left = gray_left_img[r_s: r_e, c_s: c_e]
                region_right = gray_right_img[r_s: r_e, c_s: c_e]

                power_left = self.caculate_region_gradient(region_left)
                power_right = self.caculate_region_gradient(region_right)

                if power_left > power_right:
                    fusion_l_layer[r, c, :] = left_img[r, c, :]
                else:
                    fusion_l_layer[r, c, :] = right_img[r, c, :]
        return fusion_l_layer


    def blend_lap_pyrs(self):
        result_lap_pyr = []
        gus = self.mask_gaussian_pyramid[len(self.mask_gaussian_pyramid) - 1]
        # self.result_highest_level = self.left_highest_level * gus + self.right_highest_level * (1 - gus)
        self.result_highest_level = self.fusion_highest_layer(self.left_highest_level , self.right_highest_level)
        for l in range(0 , self._levels):
            a = self.left_lap_pyr[l] * self.mask_gaussian_pyramid[l]
            b = self.right_lap_pyr[l] * (1 - self.mask_gaussian_pyramid[l])
            c = self.fusion_l_layer(a , b)
            # c = a + b
            result_lap_pyr.append(c)
        self.result_lap_pyr = np.array(result_lap_pyr)

    def build_laplacian_pyramid(self , img):
        current_img = img
        lap_pyr = []
        for l in range(0 , self._levels):
            down = cv2.pyrDown(current_img)
            up = cv2.pyrUp(down , dstsize = np.shape(current_img)[ : 2])
            lap = current_img - up
            lap_pyr.append(lap)
            current_img = down

        highest_level = current_img.copy()
        return np.array(lap_pyr) , highest_level

    def build_gaussian_pyramid(self):
        mask_gaussian_pyramid = []
        current_img = np.stack((self.blend_mask , self.blend_mask , self.blend_mask) ,axis = -1)
        mask_gaussian_pyramid.append(current_img)
        current_img = self.blend_mask
        for l in range(1 , self._levels + 1):
            if len(self.left_lap_pyr) > l:
                _down = cv2.pyrDown(current_img , dstsize = self.left_lap_pyr[l].shape[:2])
            else:
                _down = cv2.pyrDown(current_img , dstsize = self.left_highest_level.shape[:2])
            down = np.stack((_down , _down , _down) ,axis = -1)
            mask_gaussian_pyramid.append(down)
            current_img =_down
        return np.array(mask_gaussian_pyramid)

    def build_pyramids(self):
        self.left_lap_pyr , self.left_highest_level = self.build_laplacian_pyramid(self.left_img)
        self.right_lap_pyr, self.right_highest_level = self.build_laplacian_pyramid(self.right_img)
        self.mask_gaussian_pyramid = self.build_gaussian_pyramid()


def lap_fusion(img1 , img2 , m , _level):
    img1 = np.float32(img1) / 255.
    img2 = np.float32(img2) / 255


    lap = LaplacianBlending(img1, img2, m, _level)
    fusion_img = lap.blend()
    fusion_img[fusion_img > 1] = 1
    fusion_img[fusion_img < 0] = 0
    return np.uint8(fusion_img * 255)

if __name__ == '__main__':

    path = 'X:\\GXB\\20x_and_40x_data\\split_data\\'
    cz = '10140015_10000_20736'

    m = np.zeros((512, 512), dtype=np.float)
    m[:, :] = 0.5
    level = 5

    layers_fusion = []
    for layer in range(-2 , -1):
        img1 = cv2.imread(path + cz + '_' + str(layer) + '.tif')
        img2 = cv2.imread(path + cz + '_' + str(-layer) + '.tif')
        layers_fusion.append(lap_fusion(img1 , img2 , m , level))

    img_layer_0 = cv2.imread(path + cz + '_' + str(0) + '.tif')
    layers_fusion.append(img_layer_0)

    fusion5 = lap_fusion(layers_fusion[0] , layers_fusion[1] , m , level)


    cv2.imwrite('c.tif' , fusion5)
