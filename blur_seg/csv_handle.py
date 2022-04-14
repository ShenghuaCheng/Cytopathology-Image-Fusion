import numpy as np
import cv2
import os
import scipy.ndimage as ndi

def get_contours_from_csv(csv1 , csv2):

    contours_csv = []

    csv_lines1 = open(csv1, "r").readlines()
    csv_lines2 = open(csv2, "r").readlines()
    for i in range(0, len(csv_lines1)):  # 0 1
        line = csv_lines2[i]
        elems = line.strip().split(',')
        label = elems[0]
        label1 = elems[1]
        label1 = label1.strip().split(' ')
        label1 = label1[0]

        line = csv_lines1[i]
        line = line[1:(len(line) - 2)]
        elems = line.strip().split('Point:')
        if label1 == "Ellipse":
            n = len(elems)
            points = [0] * (n - 1)
            for j in range(1, n):
                s = elems[j]
                s1 = s.strip().split(',')
                x = np.round(float(s1[0]))
                y = np.round(float(s1[1]))
                points[j - 1] = [x, y]
            points = np.stack(points)
            center = (int(np.mean(points[:, 0])), int(np.mean(points[:, 1])))
            axes = (int(np.max(points[:, 0]) / 2 - np.min(points[:, 0]) / 2),
                    int(np.max(points[:, 1]) / 2 - np.min(points[:, 1]) / 2))
        elif label1 == "Polygon" or label1 == "Rectangle" or label1 == 'Area' :
            n = len(elems)
            points = [0] * (n - 1)
            for j in range(1, n):
                s = elems[j]
                s1 = s.strip().split(',')
                x = np.round(float(s1[0]))
                y = np.round(float(s1[1]))
                points[j - 1] = [x, y]

        # 进行各种类型区域的绘制
        if label in ['null']:
            if label1 == "Ellipse":
                points = cv2.ellipse2Poly(center, axes, angle=0, arcStart=0, arcEnd=360, delta=5)
            contours_csv.append(np.int32(np.stack(points)))
    return contours_csv

def read_csv():
    csv_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\fj_gxb\\'

    mask_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\combined_csv\\'

    csvs = os.listdir(csv_path)
    csvs = [x for x in csvs if x.find('.tif') == -1 and x.find('.db') == -1]
    for csv in csvs:
        print(csv)
        contours = get_contours_from_csv(csv_path + csv + '\\file1.csv' , csv_path + csv + '\\file2.csv')
        contours = np.array(contours)

        img_temp = np.zeros((1024 , 1024) , dtype = np.uint8)
        contours_new = []
        for contour in contours:
            center_y = np.mean(contour[: , 0])
            center_x = np.mean(contour[: , 1])
            img_temp[: , :] = 0
            if center_x < 512 and center_y < 512: # -2
                cv2.fillPoly(img_temp , [contour] , (255 , ))
                img_temp[0 : 512 , 512 : 1024] = 0
                img_temp[512 : 1024 , 0 : 512] = 0
            elif center_x < 512 and center_y >= 512: # 0
                cv2.fillPoly(img_temp, [contour], (255,))
                img_temp[0: 512, 0: 512] = 0
                img_temp[512: 1024, 0: 512] = 0
            elif center_x >= 512 and center_y < 512: # +2
                cv2.fillPoly(img_temp, [contour], (255,))
                img_temp[0: 512, 512: 1024] = 0
                img_temp[0: 512, 0: 512] = 0
            else:
                print('error')
                continue
            img_temp[512 :1024 , 512 : 1024] = 0
            temp_contours, _ = cv2.findContours(np.uint8(img_temp * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_new.append(temp_contours[0])

        img_temp[: , :] = 0
        cv2.fillPoly(img_temp, np.array(contours_new), (255,))


        cv2.imwrite(mask_path + csv + '_-2.tif' , img_temp[0 : 512 , 0 : 512])
        cv2.imwrite(mask_path + csv + '_0.tif' , img_temp[0 : 512 , 512 : 1024])
        cv2.imwrite(mask_path + csv + '_2.tif' , img_temp[512 : 1024 , 0 : 512])
        # cv2.imwrite(csv_path + csv + '.tif' , img_temp)


if __name__ == '__main__':
    read_csv()