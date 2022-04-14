import torch
from torch import nn
from torchvision.models.vgg import vgg16
from torch.autograd import Variable



class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self,out_labels, out_images, target_images):
        # Adversarial Loss
        valid = Variable(torch.cuda.FloatTensor(out_images.size(0),1).fill_(1.0),requires_grad = False)
        adversarial_loss = nn.BCELoss()(out_labels,valid)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return [image_loss,adversarial_loss,perception_loss,tv_loss]

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class G_loss_vgg(nn.Module):
    """
    5:第一次maxpool后
    10：第二次maxpool后
    17：第三次maxpool后
    24：第四次maxpool后
    31：第五次maxpool后
    only for single GPU
    ps:2019.1.15
    损失函数加权,只对image mse加权

    """
    def __init__(self,floor):
        super(G_loss_vgg, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:floor]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        #self.tv_loss = TVLoss()

    def forward(self,out_labels, out_images, target_images,weight_map):
        # Adversarial Loss
        valid = Variable(torch.cuda.FloatTensor(out_images.size(0),1).fill_(1.0),requires_grad = False)
        adversarial_loss = nn.BCELoss()(out_labels,valid)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        out_images = torch.mul(out_images,weight_map)
        target_images = torch.mul(target_images,weight_map)
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        #tv_loss = self.tv_loss(out_images)
        #return [image_loss,adversarial_loss,perception_loss,tv_loss]    
        return [image_loss,adversarial_loss,perception_loss]  

class PerceptionLoss(nn.Module):
    """
    5:第一次maxpool后
    10：第二次maxpool后
    17：第三次maxpool后
    24：第四次maxpool后
    31：第五次maxpool后
    """
    def __init__(self,floor):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:floor]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self,out_images, target_images):

        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        return perception_loss

class Loosen_l1(nn.Module):
    def __init__(self,ksize = 3):
        """
        Ksize:“模糊”模板的大小
        """
        super(Loosen_l1,self).__init__()
        self.l1_loss = nn.L1Loss()
        self.ksize = ksize
        self.pad = (ksize-1)//2
    def forward(self,input,output):
        pass
if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
