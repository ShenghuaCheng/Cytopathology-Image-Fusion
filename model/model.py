import math

import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch import nn
import torch
import cv2
import numpy as np
from tensorboardX import SummaryWriter
from torchsummary import summary

import os


class DenseBlockR(nn.Module):
    """
    DenseNet密连接
    """
    def __init__(self,channels,beta = 0.5):
        super(DenseBlockR,self).__init__()
        self.beta = beta
        self.conv_module1 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.conv_module2 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.conv_module3 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.conv_module4 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.last_conv = nn.Conv2d(channels,channels,3,1,padding = 1)

    def forward(self,x):
        module1_out = self.conv_module1(x)
        module1_out_temp = x+module1_out
        module2_out = self.conv_module2(module1_out_temp)
        module2_out_temp = x+module1_out_temp+module2_out
        module3_out = self.conv_module3(module2_out_temp)

        last_conv = self.last_conv(module3_out)
        out = x + last_conv*self.beta
        return out

class LightR(nn.Module):
    """
        Denseblock,Unet,大感受野
        光学层析
    """
    def __init__(self,in_c,out_c,residual_beta = 0.5):
        """
            in_c :输入的通道数
            out_c: 输出的通道数
        """
        super(LightR,self).__init__()
        self.residual_beta = residual_beta

        self.inconv = nn.Sequential(
            nn.Conv2d(in_c,64,9,1,padding=4),
            nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64,64,3,stride = 1,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlockR(64,beta = residual_beta) for _ in range(2)])
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64,128,3,stride = 1,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlockR(128,beta = residual_beta) for _ in range(2)])

        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128,256,3,stride = 1,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlockR(256,beta = residual_beta) for _ in range(2)])
        )
        self.bottom = nn.Sequential(
            nn.Conv2d(256,512,3,1,padding = 1),
            nn.PReLU(),
            nn.Conv2d(512,512,3,stride = 1,padding=1),
            nn.PReLU(),
            nn.Conv2d(512,256,3,1,padding = 1),
            nn.PReLU()
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(512,256,3,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlockR(256,beta = residual_beta) for _ in range(2)]),
            nn.Conv2d(256,128,3,padding = 1),
            nn.PReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256,128,3,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlockR(128,beta = residual_beta) for _ in range(2)]),
            nn.Conv2d(128,64,3,padding = 1),
            nn.PReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128,64,3,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlockR(64,beta = residual_beta) for _ in range(2)]),
            nn.Conv2d(64,64,3,padding = 1),
            nn.PReLU()
        )
        self.out = nn.Conv2d(64,out_c,9,1,padding = 4)
    def forward(self,x):
        cin = self.inconv(x)
        down1 = self.down1(cin)
        downsample1 = F.avg_pool2d(down1,kernel_size = 2,stride = 2)
        down2 = self.down2(downsample1)
        downsample2 = F.avg_pool2d(down2,kernel_size = 2,stride = 2)
        down3 = self.down3(downsample2)
        downsample3 = F.avg_pool2d(down3,kernel_size = 2,stride = 2)

        bottom = self.bottom(downsample3)

        upsample1 = F.interpolate(bottom,scale_factor = 2)
        cat1 = torch.cat([down3,upsample1],dim = 1)
        up1 = self.up1(cat1)
        upsample2 = F.interpolate(up1,scale_factor = 2)
        cat2 = torch.cat([down2,upsample2],dim = 1)
        up2 = self.up2(cat2)
        upsample3 = F.interpolate(up2,scale_factor = 2)
        cat3 = torch.cat([down1,upsample3],dim = 1)
        up3 = self.up3(cat3)
        out = self.out(up3)
        out = (torch.tanh(out)+1)/2
        return out


class DenseBlock(nn.Module):
    """
    DenseNet密连接
    """
    def __init__(self,channels,beta = 0.5):
        super(DenseBlock,self).__init__()
        self.beta = beta
        self.conv_module1 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.conv_module2 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.conv_module3 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.conv_module4 = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,padding=1),
                nn.LeakyReLU(inplace=True)
                )
        self.last_conv = nn.Conv2d(channels,channels,3,1,padding = 1) 
    def forward(self,x): #actually use three layer
        module1_out = self.conv_module1(x)
        module1_out_temp = x+module1_out
        module2_out = self.conv_module2(module1_out_temp)
        module2_out_temp = x+module1_out_temp+module2_out
        # module3_out = self.conv_module3(module2_out_temp)
        # module3_out_temp = x+module1_out_temp+module2_out_temp+module3_out
        # module4_out = self.conv_module4(module3_out_temp)
        module4_out_temp = x + module1_out_temp + module2_out_temp
        last_conv = self.last_conv(module4_out_temp)
        out = x + last_conv*self.beta
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

class Light(nn.Module):
    """
    Denseblock,Unet,大感受野
    光学层析
    """
    def __init__(self,in_c,out_c,residual_beta = 0.5):
        """
        in_c :输入的通道数
        out_c: 输出的通道数
        """
        super(Light,self).__init__()
        self.residual_beta = residual_beta

        self.inconv = nn.Sequential(
            nn.Conv2d(in_c,64,9,1,padding=4),
            nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64,64,3,stride = 1,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlock(64,beta = residual_beta) for _ in range(2)])
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64,128,3,stride = 1,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlock(128,beta = residual_beta) for _ in range(2)])

        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128,256,3,stride = 1,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlock(256,beta = residual_beta) for _ in range(2)])
        )
        self.bottom = nn.Sequential(
            nn.Conv2d(256,512,3,1,padding = 1),
            nn.PReLU(),
            nn.Conv2d(512,512,3,stride = 1,padding=1),
            nn.PReLU(),
            nn.Conv2d(512,256,3,1,padding = 1),
            nn.PReLU()
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(512,256,3,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlock(256,beta = residual_beta) for _ in range(2)]),
            nn.Conv2d(256,128,3,padding = 1),
            nn.PReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256,128,3,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlock(128,beta = residual_beta) for _ in range(2)]),
            nn.Conv2d(128,64,3,padding = 1),
            nn.PReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128,64,3,padding = 1),
            nn.PReLU(),
            nn.Sequential(*[DenseBlock(64,beta = residual_beta) for _ in range(2)]),
            nn.Conv2d(64,64,3,padding = 1),
            nn.PReLU()
        )
        self.out = nn.Conv2d(64,out_c,9,1,padding = 4)
    def forward(self,x):
        cin = self.inconv(x)
        down1 = self.down1(cin)
        downsample1 = F.avg_pool2d(down1,kernel_size = 2,stride = 2)
        down2 = self.down2(downsample1)
        downsample2 = F.avg_pool2d(down2,kernel_size = 2,stride = 2)
        down3 = self.down3(downsample2)
        downsample3 = F.avg_pool2d(down3,kernel_size = 2,stride = 2)

        bottom = self.bottom(downsample3)

        upsample1 = F.interpolate(bottom,scale_factor = 2)
        cat1 = torch.cat([down3,upsample1],dim = 1)
        up1 = self.up1(cat1)
        upsample2 = F.interpolate(up1,scale_factor = 2)
        cat2 = torch.cat([down2,upsample2],dim = 1)
        up2 = self.up2(cat2)
        upsample3 = F.interpolate(up2,scale_factor = 2)
        cat3 = torch.cat([down1,upsample3],dim = 1)
        up3 = self.up3(cat3)
        out = self.out(up3)
        out = (torch.tanh(out)+1)/2
        return out

class GeneratorLayers(nn.Module):
    """
        use multi layers generate fusion image
    """

    def __init__(self , layers , stage_num = 5):
        super(GeneratorLayers, self).__init__()
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(layers * 3 , 64, kernel_size=9, padding=4),
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
        self.stage2_conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size = 9,padding= 4),
            nn.PReLU()
        )
        stage2_block = [ResidualBlock(64) for _ in range(stage_num)]
        self.stage2_block = nn.Sequential(*stage2_block)
        self.stage2_conv2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size = 3,padding=1),
            nn.BatchNorm2d(64)
        )
        self.stage2_up = UpsampleBLock(64,2)
        self.stage2_conv3 = nn.Conv2d(64,3,kernel_size =9,padding=4)

    def forward(self , x):
        stage1_conv1 = self.stage1_conv1(x)
        stage1_block = self.stage1_block(stage1_conv1)
        stage1_conv2 = self.stage1_conv2(stage1_block)
        stage1_middle = torch.cat((stage1_conv1 , stage1_conv2) , dim = 1)
        stage1_middle = self.stage1_middle_conv1(stage1_middle)
        stage1_conv3 = self.stage1_conv3(stage1_middle)
        fusion20x = (torch.tanh(stage1_conv3)+1)/2

        return fusion20x

class Generator(nn.Module):
    """
    4到10再到20生成器,将9*9放在内部,生成器待配准
    """
    def __init__(self,stage1_num = 5,stage2_num = 5):
        super(Generator, self).__init__()
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        stage1_block = [ResidualBlock(64) for _ in range(stage1_num)]
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
        self.stage2_conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size = 9,padding= 4),
            nn.PReLU()
        )
        stage2_block = [ResidualBlock(64) for _ in range(stage2_num)]
        self.stage2_block = nn.Sequential(*stage2_block)
        self.stage2_conv2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size = 3,padding=1),
            nn.BatchNorm2d(64)
        )
        self.stage2_up = UpsampleBLock(64,2)
        self.stage2_conv3 = nn.Conv2d(64,3,kernel_size =9,padding=4)

    def forward(self, x ):
        """
        """
        stage1_conv1 = self.stage1_conv1(x)
        stage1_block = self.stage1_block(stage1_conv1)
        stage1_conv2 = self.stage1_conv2(stage1_block)
        stage1_middle = torch.cat((stage1_conv1 , stage1_conv2) , dim = 1)
        stage1_middle = self.stage1_middle_conv1(stage1_middle)
        # stage1_up = self.stage1_up(stage1_conv1+stage1_conv2)
        stage1_conv3 = self.stage1_conv3(stage1_middle)
        gen20x = (torch.tanh(stage1_conv3)+1)/2

        # stage2_conv1 = self.stage2_conv1(gen10)
        # stage2_block = self.stage2_block(stage2_conv1)
        # stage2_conv2 = self.stage2_conv2(stage2_block)
        # stage2_up = self.stage2_up(stage2_conv2)
        # stage2_conv3 = self.stage2_conv3(stage2_up)
        # gen20 = (torch.tanh(stage2_conv3)+1)/2

        return gen20x

class DiscriminatorPatch64(nn.Module):
    def __init__(self):
        super(DiscriminatorPatch64, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),

            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        output = self.net(x)
        output_shape = (output.size(0), output.size(2), output.size(3))
        final_output = F.sigmoid(output.view(output_shape))
        return final_output




class DiscriminatorPatch16(nn.Module):
    def __init__(self):
        super(DiscriminatorPatch16, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512 , 1 , kernel_size = 1)
            # nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(512, 1024, kernel_size=1),
            # nn.LeakyReLU(0.2),
            # nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):

        output = self.net(x)
        output_shape = (output.size(0), output.size(2), output.size(3))
        final_output = F.sigmoid(output.view(output_shape))
        return final_output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))

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

class ResidualBlock_I(nn.Module):
    """
    用InstanceNorm 代替 BN
    功能与ResidualBlock 一致
    """
    def __init__(self, channels):
        super(ResidualBlock_I, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.in1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.in2(residual)

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

#配准函数
class RegisLayer(Function):
    """
    实现在函数运行过程中进行配准
    PS:something is wrong
    """
    def forward(self,lr,hr):
        lr_ = lr.detach().cpu().numpy()
        hr_ = hr.detach().cpu().numpy()
        new10,new20 = self.regis(lr_,hr_)
        new10 = torch.from_numpy(new10).cuda()
        new20 = torch.from_numpy(new20).cuda()
        return lr.new(new10),hr.new(new20)


    def backward(self,grad_output):
        #无需改变梯度
        return grad_output

    def regis(self,ten_10x, ten_20x,single = False):
        #ten表示输入张量，single表示是否单通道配准
        ten_10x = np.transpose(ten_10x,axes=[0,2,3,1])
        ten_20x = np.transpose(ten_20x,axes=[0,2,3,1])
        ten_10x_result = np.zeros((0,250,250,3), np.float32)
        ten_20x_result = np.zeros((0,500,500,3), np.float32)
        for i in range(np.shape(ten_10x)[0]):
            img10x = ten_10x[i,:,:,:]
            img20x = ten_20x[i,:,:,:]
            img10x_250 = img10x[3:253,3:253,:]
            img10x_sample = cv2.resize(img10x_250,(500,500))
            
            if single:
                img10x_sample_gray = cv2.cvtColor(img10x_sample,cv2.COLOR_RGB2GRAY)
                img20x_gray=cv2.cvtColor(img20x,cv2.COLOR_RGB2GRAY)
            
                result=cv2.matchTemplate(img20x_gray,img10x_sample_gray,cv2.TM_CCOEFF)
                _, maxval, _, maxloc = cv2.minMaxLoc(result)
            else:
                result=cv2.matchTemplate(img20x,img10x_sample,cv2.TM_CCOEFF)
                _, maxval, _, maxloc = cv2.minMaxLoc(result)
            
            img20x_sample = img20x[maxloc[1]:(maxloc[1]+500),maxloc[0]:(maxloc[0]+500),:]
            
            ten_10x_result = np.concatenate((ten_10x_result,np.array([img10x_250])), axis=0)
            ten_20x_result = np.concatenate((ten_20x_result,np.array([img20x_sample])), axis=0)
        
        ten_10x_result = np.transpose(ten_10x_result,axes=[0,3,1,2])
        ten_20x_result = np.transpose(ten_20x_result,axes=[0,3,1,2])
        
        return ten_10x_result, ten_20x_result

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg = Discriminator().to(device)
    summary(vgg , (3 , 512 , 512))