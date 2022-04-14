import os
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import DiscriminatorPatch64
from model.model import Light
from data.dataset import Fusion20xSingleLayersTripleDataSet
from model.torch_model_resnet_aspp import ResNetASPP
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

checkpoints_path = path + 'checkpoints/fusion_0_layer_blur_mae/'
image_log_path = path + 'image_log/fusion_0_layer_blur_mae/'
tensorboard_log = path + 'run_log/fusion_0_layer_blur_mae/'
# model_save_step = 500
img_log_step = 50
batch_size = 1  # batch size
num_workers = 0
mask_threshold = 0.5
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
train_set = Fusion20xSingleLayersTripleDataSet(image_20x_data_path , image_20x_label_path , names_log , layers)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_flag,
                          num_workers=num_workers)  # 训练数据加载器
# load model
netG = Light(3 , 3).cuda()
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

#set blur seg model
model_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\middle_result\\weights_log\\44_1125.pth'
model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
model_blur_seg.load_state_dict(torch.load(model_path))
model_blur_seg.cuda()

#define preprocess data methord for model_blur_sef
def preprocess_data_blur(gen20x_label):
    blur_model_input = []
    for temp in gen20x_label:
        temp = np.uint8(temp.cpu().detach().numpy() * 255)
        temp = np.transpose(temp, [1, 2, 0])
        temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
        temp = (np.float32(temp) / 255. - 0.5) * 2
        temp = np.transpose(temp , [2 , 0 , 1])
        blur_model_input.append(temp)
    return torch.from_numpy(np.array(blur_model_input)).cuda()

netD1 = DiscriminatorPatch64().cuda()
print("# discriminator parameters(stage1)", sum(param.numel() for param in netD1.parameters()))

# set cost function
bce_loss = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()
l1_loss_blur_region = torch.nn.L1Loss()

# load pretrained model
pretrain_flag = True
if pretrain_flag:
    netG.load_state_dict(torch.load(checkpoints_path + 'netG_epoch_%d_%d.pth' % (7 , 4947)))
    netD1.load_state_dict(torch.load(checkpoints_path + 'netD_epoch_%d_%d.pth' % (7 , 4947)))

# set parallel mode
netG = torch.nn.DataParallel(netG).to(device)
netD1 = torch.nn.DataParallel(netD1).to(device)

# set network running mode
netG.train()
netD1.train()

patch_nums = 8
image_save_counter = 4948

if __name__ == "__main__":
    #######################
    #训练stage
    ######################
    # set optimizer
    lr_g = 1e-4 * 0.5
    lr_d = 1e-4 * 0.25
    optimizerD1 = optim.Adam(netD1.parameters(), lr=lr_d)
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g)

    for epoch in range(1 , stage_epoch + 1):

        if epoch == 6:
            lr_g = 1e-4 * 0.5
            lr_d = 1e-4 * 0.25
            optimizerD1 = optim.Adam(netD1.parameters(), lr=lr_d)
            optimizerG = optim.Adam(netG.parameters(), lr=lr_g)

        for i, (img20x_data , img20x_label) in enumerate(train_loader):

            if epoch < 7:
                break

            if epoch == 7 and i < img_log_step * (4947 - 4469 + 1) :
                continue

            valid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), patch_nums , patch_nums).fill_(1.0), requires_grad=False)
            invalid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), patch_nums , patch_nums).fill_(0.0), requires_grad=False)
            ############################
            # (1) Update G
            ###########################
            img20x_data = img20x_data.to(device)
            img20x_label = img20x_label.to(device)

            gen20x_label = netG(img20x_data)  #generate fusion img
            netG.zero_grad()

            blur_model_input = preprocess_data_blur(gen20x_label)
            blur_mask = model_blur_seg(blur_model_input)
            blur_mask = (blur_mask[: , 0 , : , :] > mask_threshold).float()

            # 10x监督loss
            out_labels = netD1(gen20x_label)

            # weight mse
            image_mse = l1_loss(gen20x_label, img20x_label)
            image_blur_region_mse = l1_loss_blur_region(gen20x_label * blur_mask , img20x_label * blur_mask)
            adver_loss = bce_loss(out_labels, valid)

            # 总loss
            adver_loss = 0.001 * adver_loss
            g_loss = 0.5 * image_mse + 0.5 * image_blur_region_mse + adver_loss

            g_loss.backward(retain_graph=True)
            optimizerG.step()
            ######################
            # (2) Updata D
            #####################
            # optimizer disciminator
            netD1.zero_grad()
            real_loss = bce_loss(netD1(img20x_label), valid)
            fake_loss = bce_loss(netD1(gen20x_label), invalid)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD1.step()

            if i % img_log_step == 0:
                print(np.shape(img20x_data), np.shape(img20x_label), np.shape(gen20x_label))
                img20x_data = img20x_data.cpu().detach().numpy()
                img20x_label = img20x_label.cpu().detach().numpy()
                gen20x_label = gen20x_label.cpu().detach().numpy()
                blur_mask = blur_mask[0].cpu().detach().numpy()

                img20x_data = cv2.cvtColor(np.uint8(np.transpose(img20x_data[0], [1, 2, 0]) * 255), cv2.COLOR_RGB2BGR)
                gen20x_label = cv2.cvtColor(np.uint8(np.transpose(gen20x_label[0], [1, 2, 0]) * 255), cv2.COLOR_RGB2BGR)
                img20x_label = cv2.cvtColor(np.uint8(np.transpose(img20x_label[0], [1, 2, 0]) * 255), cv2.COLOR_RGB2BGR)
                blur_mask = np.uint8(np.stack((blur_mask , blur_mask , blur_mask) , axis = -1) * 255)

                combine_img = cv2.hconcat((img20x_data , gen20x_label, blur_mask , img20x_label))

                cv2.imwrite(image_log_path + 'stage1_' + str(epoch) + '_' + str(image_save_counter) + '.tif',
                            combine_img)

                writer.add_scalar('scalar/adver_loss' , adver_loss , image_save_counter)
                writer.add_scalar('scalar/image_blur_mse' , image_blur_region_mse , image_save_counter)
                writer.add_scalar('scalar/image_mse', image_mse, image_save_counter)
                writer.add_scalar('scalar/g_loss', g_loss, image_save_counter)
                writer.add_scalar('scalar/d_loss', d_loss, image_save_counter)
                # writer.add

                torch.save(netG.module.state_dict(),
                           checkpoints_path + 'netG_epoch_%d_%d.pth' % (epoch, image_save_counter))
                torch.save(netD1.module.state_dict(),
                           checkpoints_path + 'netD_epoch_%d_%d.pth' % (epoch, image_save_counter))

                image_save_counter += 1

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
            epoch, NUM_EPOCHS, i, len(train_loader), d_loss.item(), g_loss.item()))

        lr_g = lr_g * 0.8
        lr_d = lr_d * 0.8
        for param in optimizerD1.param_groups:
            param['lr'] = lr_d
        for param in optimizerG.param_groups:
            param['lr'] = lr_g
