import os
import torch.optim as optim
from torch.utils.data import DataLoader
from compare_fusion.compare_models import Pixel2PixelGenerator
from compare_fusion.compare_models import Pixel2PixleDiscriminator
from compare_fusion.data_loader import Pixel2PixelDataLoaderMultiLayer
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import cv2
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

####config
stage_epoch = 8
NUM_EPOCHS = stage_epoch
path = 'X:/GXB/20x_and_40x_data/'
image_20x_data_path = path + 'split_data/'
image_20x_label_path = path + 'label/'
names_log = path + 'train.txt'

checkpoints_path = path + 'checkpoints/pixel2pixel_-1-1_layer/'
image_log_path = path + 'image_log/pixel2pixel_-1-1_layer/'
tensorboard_log = path + 'run_log/pixel2pixel_-1-1_layer/'
# model_save_step = 500
img_log_step = 50
lr = 0.0001  # leaning rate
batch_size = 8  # batch size
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

layers = [-1 , 0 , 1]
train_set = Pixel2PixelDataLoaderMultiLayer(image_20x_data_path , image_20x_label_path , names_log , layers)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_flag , num_workers=num_workers)

# load model
netG = Pixel2PixelGenerator(9 , 3).cuda()
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Pixel2PixleDiscriminator(9 , 3).cuda()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

# set cost function
l2_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()

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
patch_nums = 32

if __name__ == "__main__":

    # set optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=lr,betas=(0.5 , 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr * 0.125, betas=(0.5, 0.999))

    #train the netG and netD
    for epoch in range(1 , stage_epoch + 1):

        for i, (img20x_data , img20x_label) in enumerate(train_loader):

            valid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), patch_nums , patch_nums).fill_(1.0), requires_grad = False)
            invalid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), patch_nums , patch_nums).fill_(0.0), requires_grad = False)

            img20x_data = img20x_data.to(device)
            img20x_label = img20x_label.to(device)

            gen20x_label = netG(img20x_data)  # 生成融合图像

            #update discrimiantor
            netD.zero_grad()
            temp = netD(gen20x_label , img20x_data)
            loss_fake = l2_loss(netD(gen20x_label , img20x_data) , invalid)
            loss_real = l2_loss(netD(img20x_label , img20x_data) , valid)
            d_loss = loss_fake + loss_real
            d_loss.backward(retain_graph = True)
            optimizerD.step()

            #update generator
            netG.zero_grad()
            adv_loss = l2_loss(netD(gen20x_label , img20x_data) , valid)
            mae_loss = l1_loss(gen20x_label , img20x_label)
            g_loss = 0.01 * adv_loss + mae_loss

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
                img20x_data  = img20x_data[0].cpu().detach().numpy()
                img20x_label = img20x_label[0].cpu().detach().numpy()
                gen20x_label = gen20x_label[0].cpu().detach().numpy()
                temp = []
                for k in range(0 , len(layers)):
                    temp.append((np.transpose(img20x_data[k * 3 : (k + 1) * 3] , [1 , 2 , 0]) + 1) * 127.5)
                temp.append((np.transpose(gen20x_label , [1 , 2 , 0]) + 1) * 127.5)
                temp.append((np.transpose(img20x_label , [1 , 2 , 0]) + 1) * 127.5)
                img20x_data = cv2.hconcat(np.uint8(temp))
                print(np.shape(img20x_data))
                img20x_data = cv2.cvtColor(img20x_data , cv2.COLOR_RGB2BGR)

                cv2.imwrite(image_log_path + 'stage1_' + str(epoch) + '_' + str(image_save_counter) + '.tif', img20x_data)

                torch.save(netG.module.state_dict(),
                           checkpoints_path + 'netG_epoch_%d_%d.pth' % (epoch, image_save_counter))
                torch.save(netD.module.state_dict(),
                           checkpoints_path + 'netD1_epoch_%d_%d.pth' % (epoch, image_save_counter))

                writer.add_scalar('scalar/adver_loss1' , adv_loss , image_save_counter)
                writer.add_scalar('scalar/image_mse1', mae_loss , image_save_counter)
                writer.add_scalar('scalar/g_loss', g_loss, image_save_counter)
                writer.add_scalar('scalar/d_loss', d_loss, image_save_counter)

                image_save_counter += 1

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
            epoch, NUM_EPOCHS, i, len(train_loader), d_loss.item(), g_loss.item()))

        lr = lr * 0.8
        for param in optimizerD.param_groups:
            param['lr'] = lr
        for param in optimizerG.param_groups:
            param['lr'] = lr