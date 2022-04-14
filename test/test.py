import cv2
import numpy as np
class arg:

    multi_layer_path = 'D:/20x_and_40x_data/split_data'

    img_name = '10140015_10000_33792'

    pos1 = (412 , 139)
    # pos2 = (101 , 355
    pos2 = (429 , 356)

    radius = 50

    layers = [-3 , -2 , -1]


imgs = list()

for layer in arg.layers:

    cpath = '%s/%s_%d.tif' % (arg.multi_layer_path , arg.img_name , layer)
    print(cpath)
    img = cv2.imread(cpath)

    img = cv2.rectangle(img, (arg.pos1[0] - arg.radius, arg.pos1[1] - arg.radius),
                             (arg.pos1[0] + arg.radius, arg.pos1[1] + arg.radius), (0 , 0 , 255) , 2)
    img = cv2.circle(img, (arg.pos2[0], arg.pos2[1]), arg.radius, (0 , 0 , 255) , 2)
    print(np.shape(img))
    imgs.append(img)

combined = cv2.hconcat(imgs)
print(np.shape(combined))
cv2.imwrite('tmp.tif' , combined)