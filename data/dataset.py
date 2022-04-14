import torch.utils.data as data
import torch
import cv2
import os
import numpy as np
import random
class WeightDataSet(data.Dataset):
    """
    适用于4x，10x,20x超分辨，一次性读取对应的三组图
    ps:新加入weight map，所以，一次读四组图。
    """
    def __init__(self,image_4x_path,image_10x_path,image_20x_path,weight_map_path,name_log_path):
        self.image_4x_path = image_4x_path
        self.image_10x_path = image_10x_path
        self.image_20x_path = image_20x_path
        self.weight_map_path = weight_map_path
        self.name_log_path = name_log_path
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()
    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_4x_path = os.path.join(self.image_4x_path,self.image_name_list[id])
        img_10x_path = os.path.join(self.image_10x_path,self.image_name_list[id])
        img_20x_path = os.path.join(self.image_20x_path,self.image_name_list[id])
        weight_map_path = os.path.join(self.weight_map_path,self.image_name_list[id])

        img_4x = cv2.imread(img_4x_path)
        img_10x = cv2.imread(img_10x_path)
        img_20x = cv2.imread(img_20x_path)
        weight_map_20 = cv2.imread(weight_map_path,cv2.IMREAD_GRAYSCALE)/255.
        weight_map_10 = cv2.resize(weight_map_20,tuple(img_10x.shape[0:2]))
        weight_map_20 = weight_map_20.astype(np.float32)
        weight_map_10 = weight_map_10.astype(np.float32)
        weight_map_20 = np.expand_dims(weight_map_20,0)
        weight_map_10 = np.expand_dims(weight_map_10,0)
        #weight_map = np.expand_dims(weight_map,0)
        # BGR to RGB
        img_4x = cv2.cvtColor(img_4x,cv2.COLOR_BGR2RGB)
        img_10x = cv2.cvtColor(img_10x,cv2.COLOR_BGR2RGB)
        img_20x = cv2.cvtColor(img_20x,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_4x = np.transpose(img_4x, axes = (2,0,1)).astype(np.float32)/255.
        img_10x = np.transpose(img_10x, axes= (2,0,1)).astype(np.float32)/255.
        img_20x = np.transpose(img_20x, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_4x = torch.from_numpy(img_4x)
        img_10x = torch.from_numpy(img_10x)
        img_20x = torch.from_numpy(img_20x)
        weight_map_20 =torch.from_numpy(weight_map_20)
        weight_map_10 = torch.from_numpy(weight_map_10)
        return [img_4x,img_10x,img_20x,weight_map_10,weight_map_20]


class ConditionMultiLayerFusionDataSet(data.Dataset):
    def __init__(self, img_20x_data_path, img_20x_label_path, name_log_path, layers):
        self.image_20x_data_path = img_20x_data_path
        self.image_20x_label_path = img_20x_label_path
        self.name_log_path = name_log_path
        self.layers = layers
        self.image_name_list = []
        self.__read_name_log()

    def __read_name_log(self):
        with open(self.name_log_path, 'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, id):
        img_name = self.image_name_list[id][0: self.image_name_list[id].find('.tif')]

        img20x_data = []
        img20x_data_c = []
        for layer in self.layers:
            temp = cv2.imread(self.image_20x_data_path + img_name + '_' + str(layer) + '.tif')

            temp1 = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp1 = np.transpose(temp1, axes=(2, 0, 1)).astype(np.float32) / 255.
            img20x_data.append(temp1)

            temp2 = (np.transpose(temp, axes=(2, 0, 1)).astype(np.float32) / 255. - 0.5) * 2
            img20x_data_c.append(temp2)

        img20x_data = np.concatenate(img20x_data, axis=0)
        img20x_data_c = np.concatenate(img20x_data_c, axis=0)

        img20x_label_path = os.path.join(self.image_20x_label_path, img_name + '.tif')
        img20x_label = cv2.imread(img20x_label_path)

        # BGR to RGB
        img20x_label = cv2.cvtColor(img20x_label, cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img20x_label = np.transpose(img20x_label, axes=(2, 0, 1)).astype(np.float32) / 255.
        # numpy array to torch tensor
        img20x_data = torch.from_numpy(img20x_data)
        img20x_data_c = torch.from_numpy(img20x_data_c)
        img20x_label = torch.from_numpy(img20x_label)

        return [img20x_data, img20x_data_c, img20x_label]  # return data and label


class ConditionFusionDataSet(data.Dataset):
    """
            use single layer generate fusion
            example : layers = [-2 , 0 , 2]
        """

    def __init__(self, img_20x_data_path, img_20x_label_path, name_log_path, layers):
        self.image_20x_data_path = img_20x_data_path
        self.image_20x_label_path = img_20x_label_path
        self.name_log_path = name_log_path
        self.layers = layers
        self.image_name_list = []
        self.__read_name_log()

    def __read_name_log(self):
        with open(self.name_log_path, 'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, id):
        img_name = self.image_name_list[id][0: self.image_name_list[id].find('.tif')]
        r = random.randint(0, len(self.layers) - 1)

        img_20x_data_path = os.path.join(self.image_20x_data_path, img_name + '_' + str(self.layers[r]) + '.tif')
        img_20x_label_path = os.path.join(self.image_20x_label_path, img_name + '.tif')

        img_20x_label = cv2.imread(img_20x_label_path)
        img_20x_data = cv2.imread(img_20x_data_path)

        # BGR to RGB

        img_20x_label = cv2.cvtColor(img_20x_label, cv2.COLOR_BGR2RGB)

        img_20x_data_c = img_20x_data.copy()
        img_20x_data = cv2.cvtColor(img_20x_data, cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_label = np.transpose(img_20x_label, axes=(2, 0, 1)).astype(np.float32) / 255.
        img_20x_data = np.transpose(img_20x_data, axes=(2, 0, 1)).astype(np.float32) / 255.
        img_20x_data_c = np.transpose(img_20x_data_c , axes = (2 , 0 , 1)).astype(np.float32) / 255.
        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_label = torch.from_numpy(img_20x_label)
        img_20x_data_c = torch.from_numpy(img_20x_data_c)

        return [img_20x_data , img_20x_data_c ,  img_20x_label]  # 返回data , condition , label


class Fusion20xSingleLayersTripleDataSet(data.Dataset):
    """
        use single layer generate fusion
        example : layers = [-2 , 0 , 2] 
    """
    def __init__(self , img_20x_data_path , img_20x_label_path , name_log_path , layers):
        self.image_20x_data_path = img_20x_data_path
        self.image_20x_label_path = img_20x_label_path
        self.name_log_path = name_log_path
        self.layers = layers
        self.image_name_list = []
        self.__read_name_log()

    def __read_name_log(self):
        with open(self.name_log_path , 'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self , id):
        img_name = self.image_name_list[id][0 : self.image_name_list[id].find('.tif')]
        r = random.randint(0 , len(self.layers) - 1)

        img_20x_data_path = os.path.join(self.image_20x_data_path , img_name + '_' + str(self.layers[r]) + '.tif')
        img_20x_label_path = os.path.join(self.image_20x_label_path , img_name + '.tif')
        
        img_20x_label = cv2.imread(img_20x_label_path)
        img_20x_data = cv2.imread(img_20x_data_path)

        # BGR to RGB
        img_20x_label = cv2.cvtColor(img_20x_label,cv2.COLOR_BGR2RGB)
        img_20x_data = cv2.cvtColor(img_20x_data , cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_label = np.transpose(img_20x_label, axes= (2,0,1)).astype(np.float32)/255.
        img_20x_data = np.transpose(img_20x_data , axes = (2 , 0 , 1)).astype(np.float32) / 255.
        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_label = torch.from_numpy(img_20x_label)

        return [img_20x_data , img_20x_label] #返回4x，10x，20x


class Fusion20xMultiLayersTripleDataSet(data.Dataset):
    """
        use multi layers generate fusion
        example : layers = [-2 , 0 , 2]
    """
    def __init__(self , img_20x_data_path , img_20x_label_path , name_log_path , layers):
        self.image_20x_data_path = img_20x_data_path
        self.image_20x_label_path = img_20x_label_path
        self.name_log_path = name_log_path
        self.layers = layers
        self.image_name_list = []
        self.__read_name_log()

    def __read_name_log(self):
        with open(self.name_log_path , 'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self , id):
        img_name = self.image_name_list[id][0 : self.image_name_list[id].find('.tif')]
        img20x_data = []
        for layer in self.layers:
            temp = cv2.imread(self.image_20x_data_path + img_name + '_' + str(layer) + '.tif')
            temp = cv2.cvtColor(temp , cv2.COLOR_BGR2RGB)
            temp = np.transpose(temp, axes = (2,0,1)).astype(np.float32)/255.
            img20x_data.append(temp)
        img_20x_data = np.concatenate(img20x_data , axis = 0)
        
        
        img_20x_label_path = os.path.join(self.image_20x_label_path , img_name + '.tif')
        img_20x_label = cv2.imread(img_20x_label_path)

        # BGR to RGB
        img_20x_label = cv2.cvtColor(img_20x_label,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_label = np.transpose(img_20x_label, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_label = torch.from_numpy(img_20x_label)

        return [img_20x_data , img_20x_label] #返回4x，10x，20x

class Fusion20x_n2_p2_TripleDataSet(data.Dataset):
    def __init__(self,image_20x_data_path,image_20x_label_path,name_log_path , layers):
        self.image_20x_data_path = image_20x_data_path
        self.image_20x_label_path = image_20x_label_path
        self.name_log_path = name_log_path
        self.layers = layers
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()

    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_name = self.image_name_list[id][0 : self.image_name_list[id].find('.tif')]

        img20x_data = []
        for layer in self.layers:
            temp = cv2.imread(self.image_20x_data_path + img_name + '_' + str(layer) + '.tif')
            temp = cv2.cvtColor(temp , cv2.COLOR_BGR2RGB)
            temp = np.transpose(img_20x_data, axes = (2,0,1)).astype(np.float32)/255.
            img20x_data.append(temp)
            print(np.shape(img20x_data))
        img_20x_data = np.concatenate(img_20x_data , axis = 2)
        
        
        img_20x_label_path = os.path.join(self.image_20x_label_path , img_name + '.tif')
        img_20x_label = cv2.imread(img_20x_label_path)

        # BGR to RGB
        img_20x_label = cv2.cvtColor(img_20x_label,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_label = np.transpose(img_20x_label, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_label = torch.from_numpy(img_20x_label)

        return [img_20x_data , img_20x_label] #返回4x，10x，20x


class Fusion20x_0_TripleDataSet(data.Dataset):
    def __init__(self,image_20x_data_path,image_20x_label_path,name_log_path):
        self.image_20x_data_path = image_20x_data_path
        self.image_20x_label_path = image_20x_label_path
        self.name_log_path = name_log_path
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()

    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_20x_data_path = os.path.join(self.image_20x_data_path,self.image_name_list[id])
        img_20x_label_path = os.path.join(self.image_20x_label_path ,self.image_name_list[id])

        img_20x_data = cv2.imread(img_20x_data_path)
        img_20x_label = cv2.imread(img_20x_label_path)

        # BGR to RGB
        img_20x_data = cv2.cvtColor(img_20x_data,cv2.COLOR_BGR2RGB)
        img_20x_label = cv2.cvtColor(img_20x_label,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_data = np.transpose(img_20x_data, axes = (2,0,1)).astype(np.float32)/255.
        img_20x_label = np.transpose(img_20x_label, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_label = torch.from_numpy(img_20x_label)

        return [img_20x_data , img_20x_label] #返回4x，10x，20x

class TripleDataSet(data.Dataset):
    """
    适用于4x，10x,20x超分辨，一次性读取对应的三组图
    """
    def __init__(self,image_4x_path,image_10x_path,image_20x_path,name_log_path):
        self.image_4x_path = image_4x_path
        self.image_10x_path = image_10x_path
        self.image_20x_path = image_20x_path
        self.name_log_path = name_log_path
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()
    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_4x_path = os.path.join(self.image_4x_path,self.image_name_list[id])
        img_10x_path = os.path.join(self.image_10x_path,self.image_name_list[id])
        img_20x_path = os.path.join(self.image_20x_path,self.image_name_list[id])

        img_4x = cv2.imread(img_4x_path)
        img_10x = cv2.imread(img_10x_path)
        img_20x = cv2.imread(img_20x_path)

        # BGR to RGB
        img_4x = cv2.cvtColor(img_4x,cv2.COLOR_BGR2RGB)
        img_10x = cv2.cvtColor(img_10x,cv2.COLOR_BGR2RGB)
        img_20x = cv2.cvtColor(img_20x,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_4x = np.transpose(img_4x, axes = (2,0,1)).astype(np.float32)/255.
        img_10x = np.transpose(img_10x, axes= (2,0,1)).astype(np.float32)/255.
        img_20x = np.transpose(img_20x, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_4x = torch.from_numpy(img_4x)
        img_10x = torch.from_numpy(img_10x)
        img_20x = torch.from_numpy(img_20x)

        return [img_4x,img_10x,img_20x] #返回4x，10x，20x

class TestDataSet(data.Dataset):
    """
    适用于4x，10x,20x超分辨，一次性读取对应的三组图
    """
    def __init__(self,image_4x_path,image_10x_path,image_20x_path,name_log_path):
        self.image_4x_path = image_4x_path
        self.image_10x_path = image_10x_path
        self.image_20x_path = image_20x_path
        self.name_log_path = name_log_path
        self.image_name_list = []

        # init some utils function
        self.__read_name_log()
    def __read_name_log(self):
        with open(self.name_log_path,'r') as f:
            for line in f:
                name = line.strip()
                self.image_name_list.append(name)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,id):
        img_4x_path = os.path.join(self.image_4x_path,self.image_name_list[id])
        img_10x_path = os.path.join(self.image_10x_path,self.image_name_list[id])
        img_20x_path = os.path.join(self.image_20x_path,self.image_name_list[id])
 
        img_4x = cv2.imread(img_4x_path)
        img_10x = cv2.imread(img_10x_path)
        img_20x = cv2.imread(img_20x_path)

        # BGR to RGB
        img_4x = cv2.cvtColor(img_4x,cv2.COLOR_BGR2RGB)
        img_10x = cv2.cvtColor(img_10x,cv2.COLOR_BGR2RGB)
        img_20x = cv2.cvtColor(img_20x,cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_4x = np.transpose(img_4x, axes = (2,0,1)).astype(np.float32)/255.
        img_10x = np.transpose(img_10x, axes= (2,0,1)).astype(np.float32)/255.
        img_20x = np.transpose(img_20x, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor
        img_4x = torch.from_numpy(img_4x)
        img_10x = torch.from_numpy(img_10x)
        img_20x = torch.from_numpy(img_20x)

        return [img_4x,img_10x,img_20x]

def warp_image(img_x,gen_x,img_y):
    """
    将原始img_x,gen_x,img_y 组合在一起,img_x,gen_x,img_y are torch tensor
    """
    img_x = img_x.cpu().detach().numpy()
    gen_x = gen_x.cpu().detach().numpy()
    img_y = img_y.cpu().detach().numpy()
    img_x = np.transpose(img_x,axes = (1,2,0))
    gen_x = np.transpose(gen_x,axes = (1,2,0))
    img_y = np.transpose(img_y,axes = (1,2,0))
    img_x = cv2.cvtColor(img_x,cv2.COLOR_RGB2BGR)
    gen_x = cv2.cvtColor(gen_x,cv2.COLOR_RGB2BGR)
    img_y = cv2.cvtColor(img_y,cv2.COLOR_RGB2BGR)
    shape_x = img_x.shape
    shape_gen_x = gen_x.shape
    if shape_x[0] != shape_gen_x[0]:
        # 一样大小，直接组合
        img_x = cv2.resize(img_x,(shape_gen_x[0],shape_gen_x[1]))
    assemble_img = np.concatenate([img_x,gen_x,img_y],axis = 1)
    assemble_img = np.uint8(assemble_img*255)
    return assemble_img

def warp_image_plus(tensor_list):
    """
    tensor_list: list，内在元素是 pytorch tensor [c,h,w]
    for convience
    4x,gen10x,10x,gen20x,20x

    """
    # 讲tensor转为numpy 并转换通道
    assert len(tensor_list[0].shape) == 3 ,"tensor 必须 为 CHW"
    img_list = []
    for tensor in tensor_list:
        img_x = tensor.cpu().detach().numpy()
        img_x = np.transpose(img_x,axes=(1,2,0))
        img_x = cv2.cvtColor(img_x,cv2.COLOR_RGB2BGR)
        img_list.append(img_x)
    h,w,_ = img_list[-1].shape
    for i,img in enumerate(img_list[:-2]):
        img = cv2.resize(img,(h,w))
        img_list[i] = img
    assemble_img = np.concatenate(img_list,axis = 1)
    assemble_img = np.uint8(assemble_img*255)
    return assemble_img

def get_images_name(path,name_log_path,image_type = 'tif',split = True):
    """
    get image name from a dir 
    split: Ture,将其拆分为训练集和验证
    """
    import random
    from glob import glob
    path_name = os.path.join(path,'*'+image_type)
    name_list = glob(path_name)
    random.shuffle(name_list)
    if not split:
        with open(name_log_path,'w') as f:
            for name in name_list:
                name = os.path.split(name)[1]
                f.write(name)
                f.write('\n')
    else:
        path_head,file_name = os.path.split(name_log_path)
        train_path = os.path.join(path_head,'train.txt')
        test_path = os.path.join(path_head,'test.txt')
        test_list = []
        train_list = []
        for index,name in enumerate(name_list):
            if index<1000:
                test_list.append(name)
            else:
                train_list.append(name)
        with open(test_path,'w') as f:
            for name in test_list:
                name = os.path.split(name)[1]
                f.write(name)
                f.write('\n')
        with open(train_path,'w') as f:
            for name in train_list:
                name = os.path.split(name)[1]
                f.write(name)
                f.write('\n')
