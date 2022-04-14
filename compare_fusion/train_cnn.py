import os
import cv2
import numpy as np
import multiprocessing.dummy as multi
from multiprocessing import cpu_count
import torch
from tensorboardX import SummaryWriter
import random
from compare_fusion.compare_models import MFICnn


log_path = 'X:\\GXB\\paper&patent\\cnn\\train_log\\'
writer = SummaryWriter(log_path + 'run_log')

def get_item(path , cz):
    cz = cz[ : cz.find('_0.tif')]
    c_img = cv2.imread(path + 'c_img_new\\' + cz + '_0.tif' , 0)
    b_img = []
    for i in range(1 , 6):
        b_img.append(cv2.imread(path + 'b_img_new\\' + cz + '_' + str(i) + '.tif' , 0))

    c_img , b_img = np.array(c_img) , np.array(b_img)

    return [np.float32(c_img) / 255. , np.float32(b_img) / 255.]

def get_data(path , nums):

    czs = os.listdir(path + 'c_img_new\\')
    czs = [x for x in czs if x.find('.tif') != -1] #排除不是tif的图片
    c_imgs , b_imgs = [] , []

    pool= multi.Pool(cpu_count())

    for cz in czs[ : ]:
        temp = pool.apply_async(get_item , args = (path , cz)).get()
        c_imgs.append(temp[0])
        b_imgs.append(temp[1])
    pool.close()
    pool.join()

    return np.array(c_imgs) , np.array(b_imgs)

if __name__ == '__main__':
    data_path = 'X:\\GXB\\paper&patent\\cnn\\data\\'
    c_imgs , b_imgs = get_data(data_path , 1000)

    test_num = 9211
    train_c_imgs , train_b_imgs = c_imgs[test_num : ] , b_imgs[test_num : ]
    test_c_imgs, test_b_imgs = c_imgs[ : test_num], b_imgs[ : test_num]

    device = torch.device('cuda') #load cude
    cnn_model = MFICnn(1).cuda() #load model
    bce_loss = torch.nn.BCELoss() #define loss
    cnn_model = torch.nn.DataParallel(cnn_model).to(device) #parallel model
    cnn_model.train() #set train model
    optim = torch.optim.SGD(cnn_model.parameters(), lr=1e-4 , momentum = 0.9 , weight_decay = 0.0005) #set optimizer function

    train_epoch = 100
    batch_size = 128
    save_count = 0
    log_step = 20
    for epoch in range(0 , train_epoch):

        #shuffle train data and test data
        shuffle_list = [x for x in range(0 , len(train_c_imgs))]
        random.shuffle(shuffle_list)
        train_c_imgs , train_b_imgs = train_c_imgs[shuffle_list] , train_b_imgs[shuffle_list]

        shuffle_list = [x for x in range(0 , len(test_c_imgs))]
        random.shuffle(shuffle_list)
        test_c_imgs, test_b_imgs = test_c_imgs[shuffle_list], test_b_imgs[shuffle_list]


        for batch_i in range(0 , int(len(train_c_imgs) / batch_size)):
            temp_c_imgs = train_c_imgs[batch_i * batch_size : (batch_i + 1) * batch_size]
            temp_b_imgs = train_b_imgs[batch_i * batch_size : (batch_i + 1) * batch_size]

            print(np.shape(temp_b_imgs) , np.shape(temp_c_imgs))

            label = np.random.randint(0 , 2 , batch_size)
            imgs_c1 , imgs_c2 , labels = [] , [] , []
            for i in range(0 , len(label)):
                if label[i] == 0:
                    imgs_c1.append(temp_b_imgs[i , random.randint(0 , 4)])
                    imgs_c2.append(temp_c_imgs[i])
                    labels.append([0 , 1])
                else:
                    imgs_c2.append(temp_b_imgs[i, random.randint(0, 4)])
                    imgs_c1.append(temp_c_imgs[i])
                    labels.append([1 , 0])

            imgs_c1 , imgs_c2 = torch.from_numpy(np.array(imgs_c1)) , torch.from_numpy(np.array(imgs_c2))
            labels = torch.from_numpy(np.float32(labels))

            imgs_c1 , imgs_c2 , labels = imgs_c1.to(device) , imgs_c2.to(device) , labels.to(device)
            imgs_c1 = torch.reshape(imgs_c1 , (batch_size , 1 , 16 , 16))
            imgs_c2 = torch.reshape(imgs_c2 , (batch_size, 1 , 16 , 16))

            #train cnn net
            cnn_model.zero_grad()
            gen_labels= cnn_model(imgs_c1 , imgs_c2)

            loss = bce_loss(gen_labels , labels)
            loss.backward()
            optim.step()

            if save_count % log_step == 0:
                torch.save(cnn_model.module.state_dict(),
                           log_path + 'cnn_model_epoch_%d_%d.pth' % (epoch, batch_i))
                writer.add_scalar('scalar/bce_loss' , loss.item() , save_count)
            save_count += 1


            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f]" % (epoch , train_epoch , batch_i , len(train_c_imgs) / batch_size , loss.item()))