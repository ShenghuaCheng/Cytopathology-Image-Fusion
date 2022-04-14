import cv2
import random
import os
import numpy as np
import torch
import torch.utils.data as data


class Pixel2PixelDataLoaderMultiLayer(data.Dataset):
    """
        use multi layers generate fusion
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
        img20x_data = []
        for layer in self.layers:
            temp = cv2.imread(self.image_20x_data_path + img_name + '_' + str(layer) + '.tif')
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp = np.transpose(temp, axes=(2, 0, 1)).astype(np.float32) / 127.5 - 1
            img20x_data.append(temp)
        img_20x_data = np.concatenate(img20x_data, axis=0)

        img_20x_label_path = os.path.join(self.image_20x_label_path, img_name + '.tif')
        img_20x_label = cv2.imread(img_20x_label_path)

        # BGR to RGB
        img_20x_label = cv2.cvtColor(img_20x_label, cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_label = np.transpose(img_20x_label, axes=(2, 0, 1)).astype(np.float32) / 127.5 - 1
        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_label = torch.from_numpy(img_20x_label)

        return [img_20x_data, img_20x_label]  # 返回数据

class Pixel2PixelDataLoader(data.Dataset):
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
        img_20x_data = cv2.cvtColor(img_20x_data, cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_label = np.transpose(img_20x_label, axes=(2, 0, 1)).astype(np.float32) / 127.5 - 1
        img_20x_data = np.transpose(img_20x_data, axes=(2, 0, 1)).astype(np.float32) / 127.5 - 1
        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_label = torch.from_numpy(img_20x_label)

        return [img_20x_data, img_20x_label]  # return data and label

class FusionGanDataLoader(data.Dataset):
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
        img_20x_data = cv2.cvtColor(img_20x_data, cv2.COLOR_BGR2RGB)
        # H*W*C to C*H*W
        img_20x_label = np.transpose(img_20x_label, axes=(2, 0, 1)).astype(np.float32) / 127.5 - 1
        img_20x_data = np.transpose(img_20x_data, axes=(2, 0, 1)).astype(np.float32) / 127.5 - 1
        # numpy array to torch tensor
        img_20x_data = torch.from_numpy(img_20x_data)
        img_20x_label = torch.from_numpy(img_20x_label)

        return [img_20x_data, img_20x_label]  # return data and label