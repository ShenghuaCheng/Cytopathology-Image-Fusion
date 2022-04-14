'''
    将切片分成4 + 1的形式进行训练与测试
'''

import os
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import DiscriminatorPatch64
from model.model import Light
from data.dataset import ConditionFusionDataSet
from model.torch_model_resnet_aspp import ResNetASPP
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import cv2
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

####config
NUM_EPOCHS = 15 # set train epoch 10
path = 'D:/20x_and_40x_data/'
image_20x_data_path = path + 'split_data/'
image_20x_label_path = path + 'label/'
names_log = path + 'train_4.txt'
val_names_log = path + 'val_4.txt'

result_path = 'D:/20x_and_40x_data/train_log/our_blur_0_layer_continue/'
checkpoints_path = result_path + 'checkpoints/'
image_log_path = result_path + 'image_log/'  # 保存训练过程中的一部分图
summary_log_path = result_path + 'run_log'
# model_save_step = 500
img_log_step = 50
lr_g = 1e-4 * 0.5  # leaning rate
lr_d = 1e-4 * 0.25
lr_decay_ratio = 0.9
batch_size = 1  # batch size
num_workers = 0
mask_threshold = 0.5
shuffle_flag = True

if not os.path.isdir(checkpoints_path):
    os.makedirs(checkpoints_path)

if not os.path.isdir(image_log_path):
    os.makedirs(image_log_path)

writer = SummaryWriter(summary_log_path)

##############
device = torch.device('cuda')

layers = [0]
train_set = ConditionFusionDataSet(image_20x_data_path , image_20x_label_path , names_log , layers)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_flag , num_workers=num_workers)  # 训练数据加载器
# load model
netG = Light(4 , 3).cuda()
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

#set blur seg model
model_path = 'X:/GXB/20x_and_40x_data/blur_task/train_data/train_second/middle_result/weights_log/44_1125.pth'
model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99)
model_blur_seg.load_state_dict(torch.load(model_path))
model_blur_seg.cuda()

netD1 = DiscriminatorPatch64().cuda()
print("# discriminator parameters(stage1)", sum(param.numel() for param in netD1.parameters()))

# set cost function
bce_loss = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()

# load pretrained model
pretrain_flag = True
if pretrain_flag:
    pre_trained_path = r'D:\20x_and_40x_data\train_log\our_blur_0_layer\checkpoints'
    netG.load_state_dict(torch.load(pre_trained_path + '/netG_epoch_%d_%d.pth' % (10 , 5288)))
    netD1.load_state_dict(torch.load(pre_trained_path + '/netD1_epoch_%d_%d.pth' % (10 , 5288)))

# set parallel mode
netG = torch.nn.DataParallel(netG).to(device)
netD1 = torch.nn.DataParallel(netD1).to(device)

# set network running mode
netG.train()
netD1.train()

image_save_counter = 5289
patch_nums = 8

# 从验证集中取数据并且预测结果
def verification():

    with open(val_names_log , 'r') as val_file:
        names = []
        for line in val_file:
            line = line.strip()
            names.append(line)
        random_name = names[np.random.randint(0 , len(names))]

        data_name = random_name[0 : random_name.find('.tif')] + '_0.tif'
        img_20x_data = cv2.imread(image_20x_data_path + data_name)
        img_20x_label = cv2.imread(image_20x_label_path + random_name)

        # BGR to RGB
        img_20x_data_c = img_20x_data.copy()
        img_20x_data = cv2.cvtColor(img_20x_data, cv2.COLOR_BGR2RGB)
        img_20x_label = cv2.cvtColor(img_20x_label , cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_data = np.transpose(img_20x_data, axes=(2, 0, 1)).astype(np.float32) / 255.
        img_20x_data_c = np.transpose(img_20x_data_c , axes = (2 , 0 , 1)).astype(np.float32) / 255.
        img_20x_label = np.transpose(img_20x_label , axes = (2 , 0 , 1)).astype(np.float32) / 255.

        # 对图像的维度进行扩展
        img_20x_data = np.expand_dims(img_20x_data , axis = 0)
        img_20x_data_c = np.expand_dims(img_20x_data_c , axis = 0)
        img_20x_label = np.expand_dims(img_20x_label , axis = 0)

        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_data_c = torch.from_numpy(img_20x_data_c)
        img_20x_label = torch.from_numpy(img_20x_label)

        img_20x_data = img_20x_data.to(device)
        img_20x_data_c = img_20x_data_c.to(device)
        img_20x_label = img_20x_label.to(device)

        blur_mask = model_blur_seg(img_20x_data_c)
        blur_mask = (blur_mask[: , 0 , : , :] > mask_threshold).float()
        blur_mask = blur_mask.reshape((blur_mask.size(0) , 1 ,  blur_mask.size(1) , blur_mask.size(2)))

        netG_input = torch.cat([img_20x_data, blur_mask], dim = 1)

        gen20x_label = netG(netG_input)

        return img_20x_data , blur_mask , gen20x_label , img_20x_label

if __name__ == "__main__":
    #######################
    #训练stage
    ######################
    # set optimizer
    optimizerD1 = optim.Adam(netD1.parameters() , lr = lr_d)  # decrease the discriminator's lr
    optimizerG = optim.Adam(netG.parameters(), lr = lr_g)

    for epoch in range(1 , NUM_EPOCHS + 1):

        # if epoch == 6: # 不进行学习率的重置
        #     lr_g = 1e-4 * 0.5
        #     lr_d = 1e-4 * 0.25
        #     optimizerG = optim.Adam(netG.parameters(), lr=lr_g)
        #     optimizerD1 = optim.Adam(netD1.parameters(), lr=lr_d)


        for i, (img20x_data , img20x_data_c , img20x_label) in enumerate(train_loader):

            if epoch  < 11:
                break

            valid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), patch_nums , patch_nums).fill_(1.0), requires_grad=False)
            invalid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), patch_nums , patch_nums).fill_(0.0), requires_grad=False)
            ############################
            # (1) Update G
            ###########################
            img20x_data = img20x_data.to(device)
            img20x_label = img20x_label.to(device)
            img20x_data_c = img20x_data_c.to(device)

            blur_mask = model_blur_seg(img20x_data_c)
            blur_mask = (blur_mask[: , 0 , : , :] > mask_threshold).float()
            blur_mask = blur_mask.reshape((blur_mask.size(0) , 1 ,  blur_mask.size(1) , blur_mask.size(2)))

            netG_input = torch.cat([img20x_data, blur_mask], dim = 1)

            gen20x_label = netG(netG_input)  # 生成融合图像

            netG.zero_grad()
            # 10x监督loss
            out_labels = netD1(gen20x_label)
            # weight mse
            image_mse1 = l1_loss(gen20x_label, img20x_label)
            adver_loss1 = bce_loss(out_labels, valid)

            # 总loss
            adver_loss1 = 0.001 * adver_loss1
            g_loss = image_mse1 + adver_loss1

            g_loss.backward(retain_graph=True)
            optimizerG.step()

            # update netD
            netD1.zero_grad()
            real_loss = bce_loss(netD1(img20x_label), valid)
            fake_loss = bce_loss(netD1(gen20x_label), invalid)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD1.step()

            if i % img_log_step == 0:
                img_20x_data, blur_mask, gen20x_label, img_20x_label = verification()
                img20x_data = img_20x_data.cpu().detach().numpy()
                blur_mask = blur_mask.cpu().detach().numpy()
                img20x_label = img_20x_label.cpu().detach().numpy()
                gen20x_label = gen20x_label.cpu().detach().numpy()

                img20x_data = cv2.cvtColor(np.uint8(np.transpose(img20x_data[0], [1, 2, 0]) * 255), cv2.COLOR_RGB2BGR)
                gen20x_label = cv2.cvtColor(np.uint8(np.transpose(gen20x_label[0], [1, 2, 0]) * 255), cv2.COLOR_RGB2BGR)
                img20x_label = cv2.cvtColor(np.uint8(np.transpose(img20x_label[0], [1, 2, 0]) * 255), cv2.COLOR_RGB2BGR)

                contours, _ = cv2.findContours(np.uint8(blur_mask[0 , 0 , : , :] * 255), cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
                contours_img = img20x_data.copy()
                cv2.drawContours(contours_img , contours , -1 , (255 , 0 , 0) , 1)

                combine_img = cv2.hconcat((img20x_data, contours_img , gen20x_label, img20x_label))

                cv2.imwrite(image_log_path + 'stage1_' + str(epoch) + '_' + str(image_save_counter) + '.tif', combine_img)

                writer.add_scalar('scalar/adver_loss1' , adver_loss1 , image_save_counter)
                writer.add_scalar('scalar/image_mse1', image_mse1, image_save_counter)
                writer.add_scalar('scalar/g_loss', g_loss, image_save_counter)
                writer.add_scalar('scalar/d_loss', d_loss, image_save_counter)

                torch.save(netG.module.state_dict(),
                           checkpoints_path + 'netG_epoch_%d_%d.pth' % (epoch, image_save_counter))
                torch.save(netD1.module.state_dict(),
                           checkpoints_path + 'netD1_epoch_%d_%d.pth' % (epoch, image_save_counter))

                image_save_counter += 1

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
            epoch, NUM_EPOCHS, i, len(train_loader), d_loss.item(), g_loss.item()))

        lr_g = lr_g * lr_decay_ratio
        lr_d = lr_d * lr_decay_ratio
        for param in optimizerD1.param_groups:
            param['lr'] = lr_d
        for param in optimizerG.param_groups:
            param['lr'] = lr_g

