
import numpy as np

import sys
sys.path.append('/mnt/sda2/python_lib')
from fusion_metric.Metric import Metrics as me


if __name__ == '__main__':

    img_a = np.zeros((512 , 512) , dtype = np.uint8)
    img_b = np.zeros((512 , 512) , dtype = np.uint8)

    import time

    print(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    for i in range(7774):
        print(i)
        a = me.correlation_coe(img_a , img_b)
        b = me.ssim_m(img_a , img_b)
        c = me.nmi(img_a , img_b)
        print(a , b , c)

    print(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    pass