
from keras.models import Model
from keras.layers import Activation , add# , Dropout , Dense , Lambda , Input
from keras.regularizers import l2
from keras.layers import Conv2D#, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from blur_seg.BilinearUpSampling import BilinearUpSampling2D
from keras.applications.resnet50 import ResNet50
#import h5py
#自定义的卷积操作
def conv2d_bn(input_tensor, kernel_size, filters, stage, block, padding='same', strides=(1, 1), weight_decay=0., batch_momentum=0.99):
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    #卷积层，kernel_regularizer为施加在权重上的正则项，filters为滤波器
    x = Conv2D(filters, (kernel_size, kernel_size), name=conv_name_base + '2a', strides=strides, padding=padding, kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2a', momentum=batch_momentum)(x) #将上一层的结果进行批量归一化
    x = Activation('relu')(x)#激活层
    return x
##identity_block输入输出大小相同的模块
def identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay=0., batch_momentum=0.99):
    bn_axis = -1
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch' #定义卷积层的名字
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x
##conv_block输出比输入小的模块，步长增大，即会出现输出变小
def conv_block(input_tensor, kernel_size, filters, stage, block, weight_decay=0., strides=(2, 2), batch_momentum=0.99):
    bn_axis = -1
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor) #在这里增加辅助输入
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

# Atrous-Convolution version of residual blocks 空洞卷积的残差网络模块
##空洞识别模块
def atrous_identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay=0., atrous_rate=(2, 2), batch_momentum=0.99):
    bn_axis = -1
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate, padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

    x = add([x, input_tensor]) #将input_tensor与x进行相加然后做为激活层的输入,返回和，shape不变
    x = Activation('relu')(x)
    return x

##空洞卷积模块
def atrous_conv_block(input_tensor, kernel_size, filters, stage, block, weight_decay=0., strides=(1, 1), atrous_rate=(2, 2), batch_momentum=0.99):
    bn_axis = -1
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
	#filter1为输出的通道数
    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate, name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

    x = add([x, shortcut]) #取x与shortcut的加和
    x = Activation('relu')(x)
    return x
##残差网络
def ResNetASSP(input_shape = (512 , 512 , 3), weight_decay=0., batch_momentum=0.99, classes = 2 , down_sample_ratio = 8): #默认分为6个类别
    image_size = input_shape[0:2] #取出输入图像的size
    #input_shape = (960, 960, 3)
    #进行模型的定义，权重为imagenet,池化层为平均池化，include_top为是否保存顶层的3个全连接网络
    model_base = ResNet50(include_top=False, input_shape = input_shape, weights='imagenet', pooling='avg')
    
    # for layer in model_base.layers[:79]: #设置ResNet50的所有层为不可训练 37
    #     layer.trainable = False

    t = model_base.get_layer(index = 79).output
    x = model_base.get_layer("activation_40").output #获得activation_40层中间输出结果(32 * 32 * 1024)

    temp = Model(inputs = model_base.input , outputs = t)
    temp.summary()

    x = atrous_conv_block(x, 3, [256, 256, 1024], stage=11, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)
    x = atrous_identity_block(x, 3, [256, 256, 1024], stage=11, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)
    x = atrous_identity_block(x, 3, [256, 256, 1024], stage=11, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)

    x0 = conv_block(x, 1, [256, 256, 64], stage=12, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)
   
    x1 = atrous_conv_block(x, 3, [256, 256, 256], stage=13, block='a', weight_decay=weight_decay, atrous_rate=(4, 4), batch_momentum=batch_momentum)
    x1 = atrous_identity_block(x1, 3, [256, 256, 256], stage=13, block='b', weight_decay=weight_decay, atrous_rate=(4, 4), batch_momentum=batch_momentum)
    x1 = atrous_identity_block(x1, 3, [256, 256, 256], stage=13, block='c', weight_decay=weight_decay, atrous_rate=(4, 4), batch_momentum=batch_momentum)
    x1 = conv2d_bn(x1, 1, 64, stage=13, block='d')
    
    x2 = atrous_conv_block(x, 3, [256, 256, 256], stage=14, block='a', weight_decay=weight_decay, atrous_rate=(8, 8), batch_momentum=batch_momentum)
    x2 = atrous_identity_block(x2, 3, [256, 256, 256], stage=14, block='b', weight_decay=weight_decay, atrous_rate=(8, 8), batch_momentum=batch_momentum)
    x2 = atrous_identity_block(x2, 3, [256, 256, 256], stage=14, block='c', weight_decay=weight_decay, atrous_rate=(8, 8), batch_momentum=batch_momentum)
    x2 = conv2d_bn(x2, 1, 64, stage=14, block='d')
    
    x3 = atrous_conv_block(x, 3, [256, 256, 256], stage=15, block='a', weight_decay=weight_decay, atrous_rate=(12, 12), batch_momentum=batch_momentum)
    x3 = atrous_identity_block(x3, 3, [256, 256, 256], stage=15, block='b', weight_decay=weight_decay, atrous_rate=(12, 12), batch_momentum=batch_momentum)
    x3 = atrous_identity_block(x3, 3, [256, 256, 256], stage=15, block='c', weight_decay=weight_decay, atrous_rate=(12, 12), batch_momentum=batch_momentum)
    x3 = conv2d_bn(x3, 1, 64, stage=15, block='d')
    
    x = concatenate([x0, x1, x2, x3], axis=-1)
    x = conv_block(x, 3, [128, 128, 128], stage=16, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)
    x = identity_block(x, 3, [128, 128, 128], stage=16, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)
    x = identity_block(x, 3, [128, 128, 128], stage=16, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)
    x = conv2d_bn(x, 1, 16, stage=16, block='d')

    x = BilinearUpSampling2D(target_size=(image_size[0], image_size[1]))(x)
    x = conv_block(x, 3, [12, 12, 12], stage=18, block='a', weight_decay=weight_decay, strides=(1, 1),
                   batch_momentum=batch_momentum)

    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    model = Model(model_base.input, x)#得到定义好的模型并且进行返回

    return model

if __name__ == '__main__':
    model = ResNetASSP(input_shape = (512 , 512 , 3), weight_decay=0., batch_momentum=0.99, classes = 2 , down_sample_ratio = 8)

    # model_base = ResNet50(include_top=False, input_shape=(512 , 512 , 3), weights='imagenet', pooling='avg')
    model.summary()