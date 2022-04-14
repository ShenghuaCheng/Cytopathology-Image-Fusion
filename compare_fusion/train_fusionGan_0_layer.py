import os
import torch.optim as optim
from torch.utils.data import DataLoader
from compare_fusion.compare_models import FusionGanGenerator
from compare_fusion.compare_models import FusionGanDiscriminator
from compare_fusion.data_loader import FusionGanDataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import cv2
import torch
import numpy as np
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

####config
stage_epoch = 8
NUM_EPOCHS = stage_epoch
path = 'D:/20x_and_40x_data/'
image_20x_data_path = path + 'split_data/'
image_20x_label_path = path + 'label/'
names_log = path + 'train.txt'

checkpoints_path = path + 'checkpoints/fusionGan_0_layer/'
image_log_path = path + 'image_log/fusionGan_0_layer/'
tensorboard_log = path + 'run_log/fusionGan_0_layer/'
# model_save_step = 500
img_log_step = 50
lr = 0.0001  # leaning rate
batch_size = 4  # batch size
num_workers = 0
shuffle_flag = True

if not os.path.isdir(checkpoints_path):
    os.makedirs(checkpoints_path)

if not os.path.isdir(image_log_path):
    os.makedirs(image_log_path)

if not os.path.isdir(tensorboard_log):
    os.makedirs(tensorboard_log)

writer = SummaryWriter(tensorboard_log + 'run_log')

##############
device = torch.device('cuda')

layers = [0]
train_set = FusionGanDataLoader(image_20x_data_path , image_20x_label_path , names_log , layers)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_flag , num_workers=num_workers)

# load model
netG = FusionGanGenerator(3 , 3).cuda()
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = FusionGanDiscriminator(3).cuda()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

# set cost function
bce_loss = torch.nn.BCELoss()
l2_loss = torch.nn.MSELoss()

# load pretrained model
pretrain_flag = False
if pretrain_flag:
    netG.load_state_dict(torch.load(checkpoints_path + 'netG_epoch_%d_%d.pth' % (6 , 4281)))
    netD.load_state_dict(torch.load(checkpoints_path + 'netD1_epoch_%d_%d.pth' % (6 , 4281)))

# set parallel mode
netG = torch.nn.DataParallel(netG).to(device)
netD = torch.nn.DataParallel(netD).to(device)

# set network running mode
netG.train()
netD.train()

image_save_counter = 0

# get the gradient of image
def gradient(input):
    gradient_kernel = [[0. , 1. , 0.] , [1.  , -4. , 1.] , [0 , 1 , 0]]
    channel = input.size(1)
    gradient_kernel = torch.FloatTensor(gradient_kernel).expand(channel , channel , 3 , 3)
    weight = torch.nn.Parameter(data = gradient_kernel , requires_grad = False).to(device)
    return F.conv2d(input , weight , stride = 1 , padding = 1)

if __name__ == "__main__":

    # set optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=lr)
    optimizerD = optim.Adam(netD.parameters(), lr=lr)

    #train the netG and netD
    for epoch in range(1 , stage_epoch + 1):

        for i, (img20x_data , img20x_label) in enumerate(train_loader):
            min_val  , max_val = 0.7 , 1.2 # set real value
            valid = Variable(torch.rand(img20x_data.size(0) , ) * (max_val - min_val) + min_val , requires_grad = False).to(device)
            min_val , max_val = 0 , 0.3 # set fake value
            invalid = Variable(torch.rand(img20x_data.size(0) , ) * (max_val - min_val) + min_val , requires_grad = False).to(device)

            img20x_data = img20x_data.to(device)
            img20x_label = img20x_label.to(device)

            gen20x_label = netG(img20x_data)  # generate 20x's fusion image

            #update discrimiantor
            netD.zero_grad()
            loss_fake = bce_loss(netD(gen20x_label) , invalid)
            loss_real = bce_loss(netD(img20x_label) , valid)
            d_loss = loss_fake + loss_real
            d_loss.backward(retain_graph = True)
            optimizerD.step()

            #update generator
            netG.zero_grad()
            adv_loss = bce_loss(netD(img20x_data) , valid) #adv loss
            mse_loss = l2_loss(gen20x_label , img20x_label) #mse loss
            gradient_loss = l2_loss(gradient(gen20x_label) , gradient(img20x_label)) #gradient loss
            g_loss = 0.01 * adv_loss + 0.2 * mse_loss + gradient_loss

            # fix the parameters of netD
            for p in netD.parameters():
                p.requires_grad = False

            g_loss.backward()
            optimizerG.step()

            # free the parameters of netD
            for p in netD.parameters():
                p.requires_grad = True

            if i % img_log_step == 0:
                print(np.shape(img20x_data) , np.shape(img20x_label) , np.shape(gen20x_label))
                img20x_data  = img20x_data.cpu().detach().numpy()
                img20x_label = img20x_label.cpu().detach().numpy()
                gen20x_label = gen20x_label.cpu().detach().numpy()
                img20x_data = cv2.cvtColor(np.uint8((np.transpose(img20x_data[0] , [1 , 2 , 0]) + 1) * 127.5) , cv2.COLOR_RGB2BGR)
                gen20x_label = cv2.cvtColor(np.uint8((np.transpose(gen20x_label[0] , [1 , 2 , 0]) + 1) * 127.5), cv2.COLOR_RGB2BGR)
                img20x_label = cv2.cvtColor(np.uint8((np.transpose(img20x_label[0] , [1 , 2 , 0]) + 1) * 127.5), cv2.COLOR_RGB2BGR)

                combine_img = cv2.hconcat((img20x_data , gen20x_label , img20x_label))

                cv2.imwrite(image_log_path + 'stage1_' + str(epoch) + '_' + str(image_save_counter) + '.tif', combine_img)

                torch.save(netG.module.state_dict(),
                           checkpoints_path + 'netG_epoch_%d_%d.pth' % (epoch, image_save_counter))
                torch.save(netD.module.state_dict(),
                           checkpoints_path + 'netD1_epoch_%d_%d.pth' % (epoch, image_save_counter))

                writer.add_scalar('scalar/adver_loss1' , adv_loss , image_save_counter)
                writer.add_scalar('scalar/image_mse1', mse_loss , image_save_counter)
                writer.add_scalar('scalar/image_gradient_mse' , gradient_loss , image_save_counter)
                writer.add_scalar('scalar/g_loss', g_loss, image_save_counter)
                writer.add_scalar('scalar/d_loss', d_loss, image_save_counter)

                image_save_counter += 1

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
            epoch, NUM_EPOCHS, i, len(train_loader), d_loss.item(), g_loss.item()))

        # lr = lr * 0.8
        # for param in optimizerD.param_groups:
        #     param['lr'] = lr
        # for param in optimizerG.param_groups:
        #     param['lr'] = lr