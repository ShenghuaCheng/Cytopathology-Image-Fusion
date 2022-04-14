import os
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import Discriminator
from model.model import SRGANGenerator
from data.dataset import Fusion20xMultiLayersTripleDataSet
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import cv2
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

####config
stage_epoch = 8
NUM_EPOCHS = stage_epoch
path = 'X:/GXB/20x_and_40x_data/'
image_20x_data_path = path + 'split_data/'
image_20x_label_path = path + 'label/'
names_log = path + 'train.txt'

checkpoints_path = path + 'checkpoints/srgan_fusion_-1-1_layer/'
image_log_path = path + 'image_log/srgan_fusion_-1-1_layer/'
tensorboard_log = path + 'run_log/srgan_fusion_-1-1_layer/'
# model_save_step = 500
img_log_step = 50
lr = 1e-4
batch_size = 1  # batch size
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
train_set = Fusion20xMultiLayersTripleDataSet(image_20x_data_path , image_20x_label_path , names_log , layers)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_flag,
                          num_workers=num_workers)  # 训练数据加载器
# load model
netG = SRGANGenerator(9 , 5).cuda()
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD1 = Discriminator().cuda()
print("# discriminator parameters(stage1)", sum(param.numel() for param in netD1.parameters()))

# set cost function

bce_loss = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()



# load pretrained model
pretrain_flag = False  # 不加载预训练的权重
if pretrain_flag:
    netG.load_state_dict(torch.load(checkpoints_path + 'netG_epoch_%d_%d.pth' % (6 , 4281)))
    netD1.load_state_dict(torch.load(checkpoints_path + 'netD1_epoch_%d_%d.pth' % (6 , 4281)))


# set parallel mode
netG = torch.nn.DataParallel(netG).to(device)
netD1 = torch.nn.DataParallel(netD1).to(device)

# set network running mode
netG.train()
netD1.train()

image_save_counter = 0

if __name__ == "__main__":

    # set optimizer
    optimizerD1 = optim.Adam(netD1.parameters(), lr=lr * 0.5)
    optimizerG = optim.Adam(netG.parameters(), lr=lr)

    for epoch in range(1 , NUM_EPOCHS + 1):

        for i, (img20x_data , img20x_label) in enumerate(train_loader):

            valid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), 1).fill_(1.0), requires_grad=False)
            invalid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), 1).fill_(0.0), requires_grad=False)
            ############################
            # (1) Update G
            ###########################
            img20x_data = img20x_data.to(device)
            img20x_label = img20x_label.to(device)

            gen20x_label = netG(img20x_data)  # generate img

            netG.zero_grad()
            # 10x监督loss
            # print(np.shape(gen20x_label))
            out_labels = netD1(gen20x_label)
            # weight mse

            image_mse1 = l1_loss(gen20x_label, img20x_label)
            adver_loss1 = bce_loss(out_labels, valid)

            # 总loss
            adver_loss1 = 0.001 * adver_loss1
            g_loss = image_mse1 + adver_loss1

            g_loss.backward(retain_graph=True)
            optimizerG.step()
            ######################
            # (2) Updata D1
            #####################
            # 优化4x---->10x 的鉴别器 netD
            netD1.zero_grad()
            real_loss = bce_loss(netD1(img20x_label), valid)
            fake_loss = bce_loss(netD1(gen20x_label), invalid)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD1.step()

            if i % img_log_step == 0:
                print(np.shape(img20x_data) , np.shape(img20x_label) , np.shape(gen20x_label))
                img20x_data  = img20x_data[0].cpu().detach().numpy()
                img20x_label = img20x_label[0].cpu().detach().numpy()
                gen20x_label = gen20x_label[0].cpu().detach().numpy()
                temp = []
                for k in range(0 , len(layers)):
                    temp.append(np.transpose(img20x_data[k * 3 : (k + 1) * 3] , [1 , 2 , 0]) * 255)
                temp.append(np.transpose(gen20x_label , [1 , 2 , 0]) * 255)
                temp.append(np.transpose(img20x_label , [1 , 2 , 0]) * 255)
                img20x_data = cv2.hconcat(np.uint8(temp))
                print(np.shape(img20x_data))
                img20x_data = cv2.cvtColor(img20x_data , cv2.COLOR_RGB2BGR)

                cv2.imwrite(image_log_path + 'stage1_' + str(epoch) + '_' + str(image_save_counter) + '.tif', img20x_data)

                torch.save(netG.module.state_dict(),
                           checkpoints_path + 'netG_epoch_%d_%d.pth' % (epoch, image_save_counter))
                torch.save(netD1.module.state_dict(),
                           checkpoints_path + 'netD1_epoch_%d_%d.pth' % (epoch, image_save_counter))

                writer.add_scalar('scalar/adver_loss1' , adver_loss1 , image_save_counter)
                writer.add_scalar('scalar/image_mse1', image_mse1, image_save_counter)
                writer.add_scalar('scalar/g_loss', g_loss, image_save_counter)
                writer.add_scalar('scalar/d_loss', d_loss, image_save_counter)

                image_save_counter += 1

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
            epoch, NUM_EPOCHS, i, len(train_loader), d_loss.item(), g_loss.item()))

        lr = lr * 0.8
        for param in optimizerD1.param_groups:
            param['lr'] = lr
        for param in optimizerG.param_groups:
            param['lr'] = lr