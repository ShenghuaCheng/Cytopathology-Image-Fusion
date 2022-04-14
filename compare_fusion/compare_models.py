import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch import nn
import torch
import cv2
import numpy as np

class Pixel2PixelGenerator(nn.Module):

    def __init__(self , in_channel , out_channel):
        super(Pixel2PixelGenerator , self).__init__()

        self.conv2d_1 = self.conv2d(in_channel , 64 , bn = False)
        self.conv2d_2 = self.conv2d(64 , 128)
        self.conv2d_3 = self.conv2d(128 , 256)
        self.conv2d_4 = self.conv2d(256 , 512)
        self.conv2d_5 = self.conv2d(512 , 512)
        self.conv2d_6 = self.conv2d(512 , 512)

        self.conv2d_7 = self.conv2d(512 , 512)

        self.deconv2d_1 = self.deconv2d(512, 512)
        self.deconv2d_2 = self.deconv2d(1024 , 512)
        self.deconv2d_3 = self.deconv2d(1024 , 512)
        self.deconv2d_4 = self.deconv2d(1024, 256)
        self.deconv2d_5 = self.deconv2d(512, 128)
        self.deconv2d_6 = self.deconv2d(256, 64)

        self.out_layer = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2) ,
            nn.Conv2d(128, out_channel, kernel_size=3, stride=1, padding=1) ,
        )

    def conv2d(self , in_filters , out_filters, f_size = 3 , bn = True):
        """Layers used during downsampling"""
        if bn:
            return  nn.Sequential(
                nn.Conv2d(in_filters , out_filters , kernel_size = f_size , stride = 2 , padding = 1) ,
                nn.LeakyReLU() ,
                nn.BatchNorm2d(out_filters)
            )
        else:
            return  nn.Sequential(
                nn.Conv2d(in_filters , out_filters , kernel_size = f_size , stride = 2 , padding = 1) ,
                nn.LeakyReLU() ,
            )

    def deconv2d(self , in_filters , out_filters, f_size = 3, dropout_rate=0):
        """Layers used during upsampling"""
        if dropout_rate:
            return nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2) ,
                nn.Conv2d(in_filters , out_filters , kernel_size = f_size , stride = 1 , padding = 1) ,
                nn.ReLU() ,
                nn.Dropout(dropout_rate) ,
                nn.BatchNorm2d(out_filters)
            )
        else:
            return nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2) ,
                nn.Conv2d(in_filters , out_filters , kernel_size = f_size , stride = 1 , padding = 1) ,
                nn.ReLU() ,
                nn.BatchNorm2d(out_filters)
            )

    def forward(self, input):
        # Downsampling
        d1 = self.conv2d_1(input)
        d2 = self.conv2d_2(d1)
        d3 = self.conv2d_3(d2)
        d4 = self.conv2d_4(d3)
        d5 = self.conv2d_5(d4)
        d6 = self.conv2d_6(d5)

        #bottom layer
        d7 = self.conv2d_6(d6)

        # Upsampling
        u1 = torch.cat([self.deconv2d_1(d7) ,d6] , dim = 1)
        u2 = torch.cat([self.deconv2d_2(u1) , d5] , dim = 1)
        u3 = torch.cat([self.deconv2d_3(u2) , d4] , dim = 1)
        u4 = torch.cat([self.deconv2d_4(u3), d3], dim=1)
        u5 = torch.cat([self.deconv2d_5(u4), d2], dim=1)
        u6 = torch.cat([self.deconv2d_6(u5), d1], dim=1)

        output_img = torch.tanh(self.out_layer(u6))

        return output_img

class Pixel2PixleDiscriminator(nn.Module):

    def __init__(self , in_channel_A , in_channel_B):
        super(Pixel2PixleDiscriminator, self).__init__()

        self.d_layer1 = self.d_layer(in_channel_A + in_channel_B, 64, bn=False)
        self.d_layer2 = self.d_layer(64  , 128)
        self.d_layer3 = self.d_layer(128 , 256)
        self.d_layer4 = self.d_layer(256 , 512)

        self.conv2d = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)


    def d_layer(self , in_filters , out_filters, f_size = 3 , bn = True):
        """Layers used during downsampling"""
        if bn:
            return  nn.Sequential(
                nn.Conv2d(in_filters , out_filters , kernel_size = f_size , stride = 2 , padding = 1) ,
                nn.LeakyReLU() ,
                nn.BatchNorm2d(out_filters)
            )
        else:
            return  nn.Sequential(
                nn.Conv2d(in_filters , out_filters , kernel_size = f_size , stride = 2 , padding = 1) ,
                nn.LeakyReLU() ,
            )

    def forward(self, input_A , input_B):
        combined_imgs = torch.cat([input_A , input_B] , dim = 1)
        d1 = self.d_layer1(combined_imgs)
        d2 = self.d_layer2(d1)
        d3 = self.d_layer3(d2)
        d4 = self.d_layer4(d3)
        validity = self.conv2d(d4)
        return validity

class FusionGanGenerator(nn.Module):

    def __init__(self , in_channel = 3 , out_channel = 3):
        super(FusionGanGenerator , self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel , 256 , kernel_size = 5 , stride = 1 , padding = 2) ,
            nn.BatchNorm2d(256) ,
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(256 , 128 , kernel_size = 5 , stride = 1 , padding = 2) ,
            nn.BatchNorm2d(128) ,
            nn.ReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128 , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
            nn.BatchNorm2d(64) ,
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64 , 32 , kernel_size = 3 , stride = 1 , padding = 1) ,
            nn.BatchNorm2d(32) ,
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32 , out_channel , kernel_size = 1 , stride = 1)
        )

    def forward(self, input):
        layer1 = self.layer1(input)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        output = torch.tanh(layer5)
        return output

class FusionGanDiscriminator(nn.Module):

    def __init__(self , in_channel = 3):
        super(FusionGanDiscriminator , self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel , 32 , kernel_size = 3 , stride = 2) ,
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32 , 64 , kernel_size = 3 , stride = 2) ,
            nn.BatchNorm2d(64) ,
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64 , 128 , kernel_size = 3 , stride = 2) ,
            nn.BatchNorm2d(128) ,
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128 , 256 , kernel_size = 3 , stride = 2) ,
            nn.BatchNorm2d(256) ,
            nn.LeakyReLU(0.2)
        )

        self.line5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) ,
            nn.Conv2d(256 , 512 , kernel_size = 1) ,
            nn.LeakyReLU(0.2) ,
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.line5(x)
        return F.sigmoid(x.view(batch_size))

class MFICnn(nn.Module):
    """
    Multi-focus image fusion with a deep convolutional neural network
    论文中模型实现
    """
    def __init__(self , in_channel):
        super(MFICnn , self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
            nn.Conv2d(64 , 128 , kernel_size = 3 , stride = 1 , padding = 1) ,
            nn.MaxPool2d(kernel_size = 2 , stride = 2) ,
            nn.Conv2d(128 , 256 , kernel_size = 3 , stride = 1 , padding = 1)
        )

        self.fc = nn.Sequential(
            nn.Conv2d(512 , 256 , kernel_size = 8 , stride = 1) ,
            nn.Conv2d(256 , 2 , kernel_size = 1 , stride = 1)
        )

    def forward(self, img_c1 , img_c2 , test = False):

        x1 = self.cnn(img_c1)
        x2 = self.cnn(img_c2)

        x = torch.cat([x1 , x2] , dim = 1)

        out = self.fc(x)
        if not test:
            out = F.softmax(out.view((out.size(0) , out.size(1))) , dim = 1)
        else:
            out = F.softmax(out , dim=1)

        return out

class SRGANGenerator(nn.Module):
    """
        use single layer generate fusion img
    """
    def __init__(self, in_channel = 3 , stage_num = 5):
        super(SRGANGenerator, self).__init__()
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        stage1_block = [ResidualBlock(64) for _ in range(stage_num)]
        self.stage1_block = nn.Sequential(*stage1_block)
        self.stage1_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.stage1_middle_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.stage1_up = UpsampleBLock(64, 2)
        self.stage1_conv3 = nn.Conv2d(64,3,kernel_size = 9,padding= 4)

    def forward(self, x):

        stage1_conv1 = self.stage1_conv1(x)
        stage1_block = self.stage1_block(stage1_conv1)
        stage1_conv2 = self.stage1_conv2(stage1_block)
        stage1_middle = torch.cat((stage1_conv1 , stage1_conv2) , dim = 1)
        stage1_middle = self.stage1_middle_conv1(stage1_middle)
        stage1_conv3 = self.stage1_conv3(stage1_middle)
        gen20x = (torch.tanh(stage1_conv3)+1)/2

        return gen20x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x