'''
    将切片分成4 + 1的形式进行训练与测试
'''

from model.torch_model_resnet_aspp import ResNetASPP
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import DiscriminatorPatch64
from model.model import Light
from tensorboardX import SummaryWriter
from data.dataset import ConditionMultiLayerFusionDataSet
from torch.autograd import Variable
import cv2
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

####config
NUM_EPOCHS = 10
path = 'X:/GXB/20x_and_40x_data/'
image_20x_data_path = path + 'split_data/'
image_20x_label_path = path + 'label/'
names_log = path + 'train_4.txt'
val_names_log = path + 'val_4.txt'

result_path = 'D:/20x_and_40x_data/train_log/our_blur_3_layer/'
checkpoints_path = result_path + 'checkpoints/'
image_log_path = result_path + 'image_log/'  # 保存训练过程中的一部分图
summary_log_path = result_path + 'run_log'
# model_save_step = 500
img_log_step = 50
lr_g = 1e-4 * 0.5  # learning rate of g
lr_d = 1e-4 * 0.125  # learn rate of d
lr_decay_ratio = 0.9
batch_size = 1  # batch size
num_workers = 0
shuffle_flag = True

if not os.path.isdir(checkpoints_path):
    os.makedirs(checkpoints_path)

if not os.path.isdir(image_log_path):
    os.makedirs(image_log_path)

writer = SummaryWriter(summary_log_path)
##############
device = torch.device('cuda')

layers = [-1, 0, 1]
train_set = ConditionMultiLayerFusionDataSet(image_20x_data_path, image_20x_label_path, names_log, layers)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_flag,
                          num_workers=num_workers)  # 训练数据加载器
# load model , use deep net
netG = Light(12, 3).cuda()
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD1 = DiscriminatorPatch64().cuda()
print("# discriminator parameters(stage1)", sum(param.numel() for param in netD1.parameters()))

# set blur seg model
mask_threshold = 0.5
model_path = 'X:/GXB/20x_and_40x_data/blur_task/train_data/train_second/middle_result/weights_log/44_1125.pth'
model_blur_seg = ResNetASPP(classes=2, batch_momentum=0.99)
model_blur_seg.load_state_dict(torch.load(model_path))
model_blur_seg.cuda()

# set cost function

bce_loss = torch.nn.BCELoss()
l1_loss = torch.nn.L1Loss()

# load pretrained model
pretrain_flag = True  # 加载预训练的权重
if pretrain_flag:
    netG.load_state_dict(torch.load(checkpoints_path + 'netG_epoch_%d_%d.pth' % (7, 3647)))
    netD1.load_state_dict(torch.load(checkpoints_path + 'netD1_epoch_%d_%d.pth' % (7, 3647)))
# set optimizer
optimizerD1 = optim.Adam(netD1.parameters(), lr=lr_d)

# set parallel mode
netG = torch.nn.DataParallel(netG).to(device)
netD1 = torch.nn.DataParallel(netD1).to(device)

# set network running mode
netG.train()
netD1.train()

image_save_counter = 3648
patch_nums = 8


# 从验证集中取数据并且预测结果
def verification():

    with open(val_names_log , 'r') as val_file:
        names = []
        for line in val_file:
            line = line.strip()
            names.append(line)
        random_name = names[np.random.randint(0 , len(names))]

        img_name = random_name[ : random_name.find('.tif')]

        img20x_data = []
        img20x_data_c = []
        for layer in layers:
            temp = cv2.imread(image_20x_data_path + img_name + '_' + str(layer) + '.tif')

            temp1 = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp1 = np.transpose(temp1, axes=(2, 0, 1)).astype(np.float32) / 255.
            img20x_data.append(temp1)

            temp2 = (np.transpose(temp, axes=(2, 0, 1)).astype(np.float32) / 255. - 0.5) * 2
            img20x_data_c.append(temp2)

        img20x_data = np.concatenate(img20x_data, axis=0)
        img20x_data_c = np.concatenate(img20x_data_c, axis=0)

        img20x_label_path = os.path.join(image_20x_label_path, img_name + '.tif')
        img20x_label = cv2.imread(img20x_label_path)

        # BGR to RGB
        img20x_label = cv2.cvtColor(img20x_label, cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img20x_label = np.transpose(img20x_label, axes=(2, 0, 1)).astype(np.float32) / 255.
        # numpy array to torch tensor

        # 对所有的数据进行维度的扩展
        img20x_data = np.expand_dims(img20x_data , axis = 0)
        img20x_label = np.expand_dims(img20x_label , axis = 0)
        img20x_data_c = np.expand_dims(img20x_data_c , axis = 0)

        img20x_data = torch.from_numpy(img20x_data)
        img20x_data_c = torch.from_numpy(img20x_data_c)
        img20x_label = torch.from_numpy(img20x_label)

        img20x_data = img20x_data.to(device)
        img20x_label = img20x_label.to(device)

        img20x_data_c = img20x_data_c.reshape((int(img20x_data_c.size(1) / 3), 3,
                                               img20x_data_c.size(2), img20x_data_c.size(3)))
        img20x_data_c = img20x_data_c.to(device)
        blur_mask = model_blur_seg(img20x_data_c)
        blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()

        blur_mask = blur_mask.reshape((1, blur_mask.size(0), blur_mask.size(1), blur_mask.size(2)))

        netG_input = torch.cat([img20x_data, blur_mask], dim=1)

        gen20x_label = netG(netG_input)

        return img20x_data , blur_mask , gen20x_label , img20x_label


if __name__ == "__main__":
    #######################
    # 训练stage
    ######################

    optimizerG = optim.Adam(netG.parameters(), lr=lr_g)

    for epoch in range(1, NUM_EPOCHS + 1):

        # after trained 5 epoch , reset lr
        # if epoch == 6:
        #     lr_g = 1e-4 * 0.5
        #     lr_d = 1e-4 * 0.25
        #     optimizerG = optim.Adam(netG.parameters(), lr = lr_g)
        #     optimizerD1 = optim.Adam(netD1.parameters(), lr = lr_d)

        for i, (img20x_data, img20x_data_c, img20x_label) in enumerate(train_loader):

            if epoch < 7:
                break

            if epoch == 7 and i < (3647 - 3172) * img_log_step:
                print('current ', i)
                continue

            valid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), patch_nums, patch_nums).fill_(1.0),
                             requires_grad=False)
            invalid = Variable(torch.cuda.FloatTensor(img20x_data.size(0), patch_nums, patch_nums).fill_(0.0),
                               requires_grad=False)
            ############################
            # (1) Update G
            ###########################
            img20x_data = img20x_data.to(device)
            img20x_label = img20x_label.to(device)

            img20x_data_c = img20x_data_c.reshape((int(img20x_data_c.size(1) / 3), 3,
                                                   img20x_data_c.size(2), img20x_data_c.size(3)))
            img20x_data_c = img20x_data_c.to(device)
            blur_mask = model_blur_seg(img20x_data_c)
            blur_mask = (blur_mask[:, 0, :, :] > mask_threshold).float()

            blur_mask = blur_mask.reshape((1, blur_mask.size(0), blur_mask.size(1), blur_mask.size(2)))

            netG_input = torch.cat([img20x_data, blur_mask], dim=1)

            gen20x_label = netG(netG_input)

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
                img20x_data, blur_mask, gen20x_label, img20x_label = verification()
                img20x_data = img20x_data[0].cpu().detach().numpy()
                img20x_label = img20x_label[0].cpu().detach().numpy()
                gen20x_label = gen20x_label[0].cpu().detach().numpy()
                blur_mask = blur_mask.cpu().detach().numpy()
                print(np.shape(img20x_data), np.shape(img20x_label), np.shape(gen20x_label))
                multi_layer_imgs = []
                multi_layer_masks = []
                for k in range(0, len(layers)):
                    temp = np.uint8(np.transpose(img20x_data[k * 3: (k + 1) * 3], [1, 2, 0]) * 255)
                    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
                    multi_layer_imgs.append(temp)

                    contours, _ = cv2.findContours(np.uint8(blur_mask[0, k, :, :] * 255), cv2.RETR_LIST,
                                                      cv2.CHAIN_APPROX_SIMPLE)
                    temp = temp.copy()
                    cv2.drawContours(temp, contours, -1, (255, 0, 0), 1)
                    multi_layer_masks.append(temp)

                gen_img = cv2.cvtColor(np.uint8(np.transpose(gen20x_label, [1, 2, 0]) * 255), cv2.COLOR_RGB2BGR)
                label_img = cv2.cvtColor(np.uint8(np.transpose(img20x_label, [1, 2, 0]) * 255), cv2.COLOR_RGB2BGR)
                multi_layer_imgs.append(label_img)
                multi_layer_masks.append(gen_img)

                img20x_data_img = cv2.hconcat(multi_layer_imgs)
                img20x_data_mas = cv2.hconcat(multi_layer_masks)
                img = cv2.vconcat((img20x_data_img, img20x_data_mas))

                cv2.imwrite(image_log_path + 'stage1_' + str(epoch) + '_' + str(image_save_counter) + '.tif', img)

                writer.add_scalar('scalar/adver_loss1', adver_loss1, image_save_counter)
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