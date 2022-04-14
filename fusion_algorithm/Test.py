import pywt
import cv2
import numpy as np

def wt_fusion_gray(img1 , img2):
    # Params
    FUSION_METHOD = 'max'  # Can be 'min' || 'max || anything you choose according theory
    # First: Do wavelet transform on each image
    wavelet = 'db1'
    cooef1 = pywt.wavedec2(img1[:, :], wavelet)
    cooef2 = pywt.wavedec2(img2[:, :], wavelet)

    # Second: for each level in both image do the fusion according to the desire option
    fusedCooef = []
    for i in range(len(cooef1) - 1):

        # The first values in each decomposition is the apprximation values of the top level
        if (i == 0):

            fusedCooef.append(fuseCoeff(cooef1[0], cooef2[0], FUSION_METHOD))

        else:

            # For the rest of the levels we have tupels with 3 coeeficents
            c1 = fuseCoeff(cooef1[i][0], cooef2[i][0], FUSION_METHOD)
            c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
            c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)

            fusedCooef.append((c1, c2, c3))

    # Third: After we fused the cooefficent we nned to transfor back to get the image
    fusedImage = pywt.waverec2(fusedCooef, wavelet)
    # Forth: normmalize values to be in uint8
    fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage), (np.max(fusedImage) - np.min(fusedImage))), 255)
    fusedImage = fusedImage.astype(np.uint8)
    return fusedImage

def wt_fusion_rgb(img1 , img2):
    print(np.shape(img1) , np.shape(img2))
    fused_img_r = wt_fusion_gray(img1[: , : , 0] , img2[: , : , 0])
    fused_img_g = wt_fusion_gray(img1[:, :, 1], img2[:, :, 1])
    fused_img_b = wt_fusion_gray(img1[:, :, 2], img2[:, :, 2])
    fused_img = np.stack([fused_img_r , fused_img_g , fused_img_b] , axis = -1)

    return fused_img


# This function does the coefficient fusing according to the fusion method
def fuseCoeff(cooef1, cooef2, method):

    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []

    return cooef


if __name__ == '__main__':

    path = 'X:\\GXB\\20x_and_40x_data\\split_data\\'

    # Read the two image
    I1 = cv2.imread(path + '10140015_10000_12672_0.tif')
    I2 = cv2.imread(path + '10140015_10000_12672_-1.tif')

    fusedImage = wt_fusion_rgb(I1 , I2)


    # Fith: Show image
    cv2.imwrite('a.tif' , fusedImage)
