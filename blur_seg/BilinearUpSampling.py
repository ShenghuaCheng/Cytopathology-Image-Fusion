import keras.backend as K
import tensorflow as tf
from keras.layers import *
import numpy as np

def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default': #如果没有闯入数据存储的格式，通过表达式显示获取数据格式
        data_format = K.image_data_format() #得到数据表示格式
    if data_format == 'channels_first':
        original_shape = K.int_shape(X) #返回张量或变量的尺寸
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32')) #定义新的数据shape
        else:
            new_shape = tf.shape(X)[2:] #(batch_size , channels , width , height)
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32')) #否则根据放缩因子进行新shape的构建
        X = K.permute_dimensions(X, [0, 2, 3, 1])#按照给定的模式重排一个张量的轴
        X = tf.image.resize_bilinear(X, new_shape) #调整X为new_shape,使用双线性插值
        X = K.permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X #重新设置好图像shape后进行返回
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]#(batch_size , width , height , channels)
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format) #返回格式非法错误

class BilinearUpSampling2D(Layer): #二维图像的线性上采样
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs): #初始化
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs) #调用父类进行初始化

    def compute_output_shape(self, input_shape): #计算输出shape，这个函数没有用到
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height) #进行输出的返回
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None): #得到上采样后的图像
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self): #获得配置，似乎没用到
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
