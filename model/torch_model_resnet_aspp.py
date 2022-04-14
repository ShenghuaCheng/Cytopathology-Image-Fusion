import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torchvision
import torch
import numpy as np
from torchviz import make_dot , make_dot_from_trace
from tensorboardX import SummaryWriter

class Conv2d_BN(nn.Module):
    def __init__(self , in_channels , out_channels , kernel_size , strides = (1 , 1) , batch_momentum = 0.99):
        super(Conv2d_BN , self).__init__()

        self.conv2d_bn_module = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , kernel_size , strides , padding = int(kernel_size / 2)) ,
            nn.BatchNorm2d(out_channels , momentum = batch_momentum) ,
            nn.ReLU()
        )

    def forward(self, input):
        output = self.conv2d_bn_module(input)
        return output

class Identity_Block(nn.Module):
    def __init__(self , in_channels , out_channels , kernel_size , batch_momentum = 0.99):
        super(Identity_Block, self).__init__()
        out_channel1 , out_channel2 , out_channel3 = out_channels

        self.con2d_bn_module1 = nn.Sequential(
            nn.Conv2d(in_channels , out_channel1 , kernel_size = 1 , stride = 1) ,
            nn.BatchNorm2d(out_channel1 , momentum = batch_momentum) ,
            nn.ReLU()
        )

        self.con2d_bn_module2 = nn.Sequential(
            nn.Conv2d(out_channel1 , out_channel2 , kernel_size = kernel_size , stride = 1 , padding = int(kernel_size / 2)) ,
            nn.BatchNorm2d(out_channel2 , momentum = batch_momentum) ,
            nn.ReLU()
        )

        self.con2d_bn_module3 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel3, kernel_size=kernel_size, stride=1, padding = int(kernel_size / 2)),
            nn.BatchNorm2d(out_channel3, momentum=batch_momentum),
        )

        self.relu = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, input):
        x = self.con2d_bn_module1(input)
        x = self.con2d_bn_module2(x)
        x = self.con2d_bn_module3(x)
        x = self.relu(input + x)
        return x

class Conv_Block(nn.Module):
    def __init__(self , in_channels , out_channels , kernel_size , stride = 2 , batch_momentum = 0.99):
        super(Conv_Block, self).__init__()
        out_channel1 , out_channel2 , out_chnnel3 = out_channels

        self.conv2d_bn_module1 = nn.Sequential(
            nn.Conv2d(in_channels , out_channel1 , kernel_size = 1 , stride = stride) ,
            nn.BatchNorm2d(out_channel1 , momentum = batch_momentum) ,
            nn.ReLU()
        )

        self.conv2d_bn_module2 = nn.Sequential(
            nn.Conv2d(out_channel1 , out_channel2 , kernel_size = kernel_size , stride = 1 , padding = int(kernel_size / 2)) ,
            nn.BatchNorm2d(out_channel2 , momentum = batch_momentum) ,
            nn.ReLU()
        )

        self.conv2d_module3 = nn.Sequential(
            nn.Conv2d(out_channel2 , out_chnnel3 , kernel_size = 1 , stride = 1) ,
            nn.BatchNorm2d(out_chnnel3 , momentum = batch_momentum) ,
        )

        self.conv2d_module4 = nn.Sequential(
            nn.Conv2d(in_channels , out_chnnel3 , kernel_size = 1 , stride = stride) ,
            nn.ReLU()
        )
        self.relu = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, input):
        x = self.conv2d_bn_module1(input)
        x = self.conv2d_bn_module2(x)
        x = self.conv2d_module3(x)

        shortcout = self.conv2d_module4(input)
        x = self.relu(x + shortcout)

        return x

class Atrous_Identity_Block(nn.Module):
    def __init__(self , in_channels , out_channels , kernel_size , atrous_rate = 2 , batch_momentum = 0.99):
        super(Atrous_Identity_Block, self).__init__()
        out_channel1 , out_channel2 , out_channel3 = out_channels

        self.conv2d_bn_module1 = nn.Sequential(
            nn.Conv2d(in_channels , out_channel1 , kernel_size = 1 , stride = 1) ,
            nn.BatchNorm2d(out_channel1 , momentum = batch_momentum) ,
            nn.ReLU()
        )

        self.atrous_conv2d_bn_module2 = nn.Sequential(
            nn.Conv2d(out_channel1 , out_channel2 , kernel_size = kernel_size , dilation = atrous_rate , padding = int(kernel_size / 2) * atrous_rate) ,
            nn.BatchNorm2d(out_channel2 , momentum = batch_momentum) ,
            nn.ReLU()
        )

        self.conv2d_bn_module3 = nn.Sequential(
            nn.Conv2d(out_channel2 , out_channel3 , kernel_size = 1 , stride = 1) ,
            nn.BatchNorm2d(out_channel3 , momentum = batch_momentum)
        )

        self.relu = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, input):
        x = self.conv2d_bn_module1(input)
        x = self.atrous_conv2d_bn_module2(x)
        x = self.conv2d_bn_module3(x)
        x = self.relu(x + input)
        return x

class Atrous_Conv_Block(nn.Module):

    def __init__(self , in_channels , out_channels , kernel_size , stride = 1 , atrous_rate = 2 , batch_momentum = 0.99):
        super(Atrous_Conv_Block, self).__init__()
        out_channel1 , out_channel2 , out_channel3 = out_channels

        self.conv2d_bn_module1 = nn.Sequential(
            nn.Conv2d(in_channels , out_channel1 , kernel_size = 1 , stride = stride) ,
            nn.BatchNorm2d(out_channel1 , momentum = batch_momentum) ,
            nn.ReLU()
        )

        self.atrous_conv2d_bn_module2 = nn.Sequential(
            nn.Conv2d(out_channel1 , out_channel2 , kernel_size = kernel_size , stride = 1 , dilation = atrous_rate , padding = int(kernel_size / 2) * atrous_rate) ,
            nn.BatchNorm2d(out_channel2 , momentum = batch_momentum) ,
            nn.ReLU()
        )

        self.conv2d_bn_module3 = nn.Sequential(
            nn.Conv2d(out_channel2 , out_channel3 , kernel_size = 1 , stride = 1) ,
            nn.BatchNorm2d(out_channel3 , momentum = batch_momentum)
        )

        self.conv2d_bn_module4 = nn.Sequential(
            nn.Conv2d(in_channels , out_channel3 , kernel_size = 1 , stride = stride) ,
            nn.BatchNorm2d(out_channel3 , momentum = batch_momentum)
        )
        self.relu = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, input):
        x = self.conv2d_bn_module1(input)
        x = self.atrous_conv2d_bn_module2(x)
        x = self.conv2d_bn_module3(x)

        shortcut = self.conv2d_bn_module4(input)
        x = self.relu(x + shortcut)

        return x

class ResNetASPP(nn.Module):

    def __init__(self , classes , batch_momentum):
        super(ResNetASPP , self).__init__()
        base_model = torchvision.models.resnet50(pretrained = False)
        self.resnet_layer = nn.Sequential(*list(base_model.children())[:-3])

        self.stage_11_a_atrous_conv_block = nn.Sequential(
            *[Atrous_Conv_Block(1024 , [256 , 256 , 1024] , kernel_size = 3 , stride = 1 , atrous_rate = 2 , batch_momentum = batch_momentum)]
        )
        self.stage_11_b_atrous_identity_block = nn.Sequential(
            *[Atrous_Identity_Block(1024 , [256 , 256 , 1024] , kernel_size = 3 , atrous_rate = 2 , batch_momentum = batch_momentum)]
        )
        self.stage_11_c_atrous_identity_block = nn.Sequential(
            *[Atrous_Identity_Block(1024, [256, 256, 1024], kernel_size=3, atrous_rate=2, batch_momentum=batch_momentum)]
        )

        self.stage_12_a_conv_block = nn.Sequential(
            *[Conv_Block(1024 , [256 , 256 , 64] , kernel_size = 1 , stride = 1 , batch_momentum = batch_momentum)]
        )

        self.stage_13_a_atrous_conv_block = nn.Sequential(
            *[Atrous_Conv_Block(1024 , [256 , 256 , 256] , kernel_size = 3 , stride = 1 , atrous_rate = 4 , batch_momentum = batch_momentum)]
        )
        self.stage_13_b_atrous_identity_block = nn.Sequential(
            *[Atrous_Identity_Block(256, [256, 256, 256], kernel_size=3, atrous_rate=4, batch_momentum=batch_momentum)]
        )
        self.stage_13_c_atrous_identity_block = nn.Sequential(
            *[Atrous_Identity_Block(256, [256, 256, 256], kernel_size=3, atrous_rate=4, batch_momentum=batch_momentum)]
        )
        self.stage_13_d_conv_bn_block = nn.Sequential(
            *[Conv2d_BN(256 , 64 , kernel_size = 1 , strides = 1 , batch_momentum = batch_momentum)]
        )

        self.stage_14_a_atrous_conv_block = nn.Sequential(
            *[Atrous_Conv_Block(1024 , [256 , 256 , 256] , kernel_size = 3, stride = 1 , atrous_rate = 8 , batch_momentum = batch_momentum)]
        )
        self.stage_14_b_atrous_identity_block = nn.Sequential(
            *[Atrous_Identity_Block(256 , [256 , 256 , 256] , kernel_size = 3 , atrous_rate = 8 , batch_momentum = batch_momentum)]
        )
        self.stage_14_c_atrous_identity_block = nn.Sequential(
            *[Atrous_Identity_Block(256, [256, 256, 256], kernel_size=3, atrous_rate=8, batch_momentum=batch_momentum)]
        )
        self.stage_14_d_conv2d_bn = nn.Sequential(
            *[Conv2d_BN(256 , 64 , kernel_size = 1 , strides = 1 , batch_momentum = batch_momentum)]
        )

        self.stage_15_a_atrous_conv_block = nn.Sequential(
            *[Atrous_Conv_Block(1024 , [256 , 256 , 256] , kernel_size = 3, stride = 1 , atrous_rate = 12 , batch_momentum = batch_momentum)]
        )
        self.stage_15_b_atrous_identity_block = nn.Sequential(
            *[Atrous_Identity_Block(256 , [256 , 256 , 256] , kernel_size = 3 , atrous_rate = 12 , batch_momentum = batch_momentum)]
        )
        self.stage_15_c_atrous_identity_block = nn.Sequential(
            *[Atrous_Identity_Block(256, [256, 256, 256], kernel_size=3, atrous_rate=12, batch_momentum=batch_momentum)]
        )
        self.stage_15_d_conv2d_bn = nn.Sequential(
            *[Conv2d_BN(256 , 64 , kernel_size = 1 , strides = 1 , batch_momentum = batch_momentum)]
        )

        self.stage_16_a_conv_block = nn.Sequential(
            *[Conv_Block(256 , [128 , 128 , 128] , kernel_size = 3 , stride = 1 , batch_momentum = batch_momentum)]
        )
        self.stage_16_b_identity_block = nn.Sequential(
            *[Identity_Block(128 , [128 , 128 , 128] , kernel_size = 3 , batch_momentum = batch_momentum)]
        )
        self.stage_16_c_identity_block = nn.Sequential(
            *[Identity_Block(128 , [128 , 128 , 128] , kernel_size = 3 , batch_momentum = batch_momentum)]
        )
        self.stage_16_d_conv2d_bn = nn.Sequential(
            *[Conv2d_BN(128 , 16 , kernel_size = 1, strides = 1 , batch_momentum = batch_momentum)]
        )

        self.up_sample = nn.Sequential(
            nn.UpsamplingBilinear2d(size = (512 , 512))
        )

        self.stage_17_a_conv_block = nn.Sequential(
            *[Conv_Block(16, [12, 12, 12], kernel_size=3, stride=1, batch_momentum=batch_momentum)]
        )

        self.stage_17_b_conv2d_soft = nn.Sequential(
            nn.Conv2d(12 , classes , kernel_size = 1) ,
            nn.Softmax(dim = 1)
        )

    def forward(self, input):
        x = self.resnet_layer(input)
        x = self.stage_11_a_atrous_conv_block(x)
        x = self.stage_11_b_atrous_identity_block(x)
        x = self.stage_11_c_atrous_identity_block(x)

        x0 = self.stage_12_a_conv_block(x)

        x1 = self.stage_13_a_atrous_conv_block(x)
        x1 = self.stage_13_b_atrous_identity_block(x1)
        x1 = self.stage_13_c_atrous_identity_block(x1)
        x1 = self.stage_13_d_conv_bn_block(x1)

        x2 = self.stage_14_a_atrous_conv_block(x)
        x2 = self.stage_14_b_atrous_identity_block(x2)
        x2 = self.stage_14_c_atrous_identity_block(x2)
        x2 = self.stage_14_d_conv2d_bn(x2)

        x3 = self.stage_15_a_atrous_conv_block(x)
        x3 = self.stage_15_b_atrous_identity_block(x3)
        x3 = self.stage_15_c_atrous_identity_block(x3)
        x3 = self.stage_15_d_conv2d_bn(x3)

        x = torch.cat([x0 , x1 , x2 , x3] , dim = 1)

        x = self.stage_16_a_conv_block(x)
        x = self.stage_16_b_identity_block(x)
        x = self.stage_16_c_identity_block(x)
        x = self.stage_16_d_conv2d_bn(x)

        x = self.up_sample(x)  #interpolate upsample

        x = self.stage_17_a_conv_block(x)
        x = self.stage_17_b_conv2d_soft(x)

        return x

if __name__ == '__main__':
    dummy_input = Variable(torch.rand(1, 3, 512, 512))  #
    model = ResNetASPP(classes = 2 , batch_momentum = 0.99)
    out_put = model(dummy_input)