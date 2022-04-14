import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.optim as optim
from model.torch_model_resnet_aspp import ResNetASPP
from blur_seg.read_data import DataRead
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.autograd import Variable
import cv2
import torch
import random
import numpy as np

NUM_EPOCH = 50
save_middle_log = 25

lr = 1e-3
batch_size = 4

device = torch.device('cuda')
model_blur_seg = ResNetASPP(classes = 2 , batch_momentum = 0.99).cuda()

bce_loss = torch.nn.BCELoss()

pretrain_flag = False
if pretrain_flag:
    model_blur_seg.load_state_dict()

optimizer_blur = optim.Adam(model_blur_seg.parameters() , lr = lr , weight_decay = 0)

#set parallel mode
model_blur_seg = torch.nn.DataParallel(model_blur_seg).to(device)

#set network running mode
model_blur_seg.train()


def save_middle_result(middle_result , imgs , masks , epoch , batch_i):
    imgs = imgs.to(device)
    pre_masks = model_blur_seg(imgs)

    imgs = imgs.cpu().detach().numpy()
    pre_masks = pre_masks.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()

    imgs = np.transpose(imgs , [0 , 2 , 3 , 1])
    pre_masks = np.transpose(pre_masks, [0, 2, 3, 1])
    masks = np.transpose(masks, [0, 2, 3, 1])


    for k in range(0 , len(imgs)):

        pre_mask = pre_masks[k , : , : , 0] > 0.5
        # pre_mask = ndi.binary_fill_holes(pre_mask) #no need to fill holes

        contours , _ = cv2.findContours(np.uint8(pre_mask * 255) , cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
        img = np.uint8((imgs[k] / 2 + 0.5) * 255)
        img = img.copy()
        cv2.drawContours(img , contours , -1 , (255 , 0 , 0) , 2)

        mask = np.uint8(masks[k , : , : , 0] * 255)
        pre_mask = np.uint8(pre_mask * 255)
        mask = np.stack([mask , mask , mask] , axis = 2)
        pre_mask = np.stack([pre_mask , pre_mask , pre_mask] , axis = 2)

        combined_img = cv2.hconcat((img , pre_mask , mask))

        cv2.imwrite(middle_result + 'imgs_log\\' + str(epoch) + '_' + str(batch_i) + '_' + str(k) + '.tif' , combined_img)

    torch.save(model_blur_seg.module.state_dict(),
               middle_result + 'weights_log\\' + str(epoch) + '_' + str(batch_i) + '.pth')

if __name__ == '__main__':
    train_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\train\\'
    test_path = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\test\\'

    middle_result = 'X:\\GXB\\20x_and_40x_data\\fusion_mask_task\\train_data\\train_second\\middle_result\\'

    dr = DataRead(train_path, test_path)

    train_imgs, train_masks = dr.read_data(train_test_flag=0 , torch_keras_flag = 1)
    test_imgs, test_masks = dr.read_data(train_test_flag=1 , torch_keras_flag = 1)

    train_list = [x for x in range(0, len(train_imgs))]
    test_list = [x for x in range(0, len(test_imgs))]

    print(np.shape(train_imgs), np.shape(test_imgs))

    for epoch in range(1, NUM_EPOCH + 1):
        random.shuffle(train_list)

        train_imgs, train_masks = train_imgs[train_list], train_masks[train_list]

        for batch_i in range(0, int(len(train_imgs) / batch_size)):
            temp_imgs = train_imgs[batch_i * batch_size: (batch_i + 1) * batch_size]
            temp_masks = train_masks[batch_i * batch_size: (batch_i + 1) * batch_size]


            temp_imgs = temp_imgs.to(device)
            temp_masks = temp_masks.to(device)

            gen_temp_masks = model_blur_seg(temp_imgs)
            model_blur_seg.zero_grad()
            print(np.shape(gen_temp_masks) , np.shape(temp_masks))
            loss_fore = bce_loss(gen_temp_masks[: , 0 , : , :] , temp_masks[: , 0 , : , :])
            loss_back = bce_loss(gen_temp_masks[: , 1 , : , :] , temp_masks[: , 1 , : , :])
            loss = loss_fore + 0.5 * loss_back
            loss.backward(retain_graph = True)
            optimizer_blur.step()


            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]" % (epoch, NUM_EPOCH,
                                                                             batch_i, int(len(train_imgs) / batch_size),
                                                                             loss.item()))

            random.shuffle(test_list)
            if batch_i % save_middle_log == 0:  # every 25 batch save middle result
                save_middle_result(middle_result, test_imgs[test_list[0: 4]], test_masks[test_list[0: 4]],
                                   epoch, batch_i)

