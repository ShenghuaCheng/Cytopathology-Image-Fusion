import numpy as np
import cv2


img_shape = (512 , 512 , 3)

#简单加权融合算法
def wa(imgs):
    fusion_img = np.zeros(img_shape , dtype = np.uint8)
    fusion_img[: , : , 0] = np.sum(imgs[: , : , : , 0] , axis = 0) / len(imgs)
    fusion_img[:, :, 1] = np.sum(imgs[:, :, :, 1], axis=0) / len(imgs)
    fusion_img[:, :, 2] = np.sum(imgs[:, :, :, 2], axis=0) / len(imgs)
    return fusion_img

if __name__ == '__main__':
    imgs = []
    path = 'X:\\GXB\\20x_and_40x_data\\split_data\\'
    cz = '10140015_10000_13824'
    layers = [-5 , -4 , -3 , -2 , -1 , 0 , 1 , 2 , 3 , 4 , 5]
    for layer in layers:
        img = cv2.imread(path + cz + '_' + str(layer) + '.tif')
        imgs.append(img)
    imgs = np.array(imgs)

    cv_fusion_img = cv2.imread('X:\\GXB\\20x_and_40x_data\\label\\' + cz + '.tif')

    fusion_img = wa(imgs)
    combined_img = cv2.hconcat((fusion_img , cv_fusion_img))
    cv2.imwrite('fu.tif' , combined_img)
