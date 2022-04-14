import cv2
import numpy as np

img_names= ['10140015_22288_59904' , '10140015_31504_18816' , '10140015_27664_15360']

points = np.array([
                    [[[154 , 80] , [220 , 80] , [220 , 146] , [154 , 146]]] ,
                    [[[274 , 349] , [416 , 349] , [416 , 491] , [274 , 491]]] ,
                    [[[347 , 147] , [400 , 147] , [400 , 200] , [347 , 200]]]
                   ])

path = r'X:\GXB\20x_and_40x_data\test_result\compares\gen_0_fusion'
result_path = r'X:\GXB\paper&patent\paper_img\temp_imgs4'

cs = ['ori_0_layer' , 'unet_catblur_patch64' , 'label']

# for name, p in zip(img_names , points):
#
#     for c in cs:
#         cname = name
#         if c == 'ori_0_layer':
#             cname += '_0'
#         cname += '.tif'
#         img = cv2.imread(path + '\\' + c + '\\' + cname)
#         img = np.array(img)
#         img = img[p[0 , 0 , 1] : p[0 , 2 , 1] , p[0 , 0 , 0] : p[0 , 1 , 0]]
#
#         img = cv2.resize(img , (256 , 256))
#         cv2.imwrite(result_path + '\\' + name + '_' + c + '.tif' , img)


#draw_contours
s_points = np.array([[22288 , 59904] , [31504 ,18816] , [27664 , 15360]])
img = np.array(cv2.imread(r'X:\GXB\20x_and_40x_data\R\final_img.tif'))

for i in range(0 , len(s_points)):

    for j in range(0 , len(points[i , 0])):
        points[i , 0 , j , 0] = (points[i , 0 , j , 0] + s_points[i , 0]) // 10
        points[i, 0, j, 1] = (points[i, 0, j, 1] + s_points[i, 1]) // 10

print(points)

imgc = img.copy()

cv2.drawContours(imgc , points , -1 , (0 , 0 , 255) , 3)
cv2.imwrite(r'X:\GXB\20x_and_40x_data\R\final_img_c.tif' , imgc)