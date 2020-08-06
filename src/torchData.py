import torch
import torch.nn as nn	# 各种层类型的实现
import torch.nn.functional as F	# 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim	# 实现各种优化算法的包
from torchvision import datasets, transforms


import torch.utils.data as Data
import random
 
class Dataset(Data.Dataset):
    def __init__(self,file_path,nraws,shuffle=False):
        """
        file_path: the path to the dataset file
        nraws: each time put nraws sample into memory for shuffle
        shuffle: whether the data need to shuffle
        """
        file_raws = 0 
        # get the count of all samples
        with open(file_path,'r') as f:
            for _ in f:
                file_raws+=1
        self.file_path = file_path
        self.file_raws = file_raws
        self.nraws = nraws
        self.shuffle = shuffle
 
    def initial(self):
        self.finput = open(self.file_path,'r')
        self.samples = list()
 
        # put nraw samples into memory
        for _ in range(self.nraws):
            data = self.finput.readline()   # data contains the feature and label
            if data:
                self.samples.append(data)
            else:
                break
        self.current_sample_num = len(self.samples)
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.samples)
 
    def __len__(self):
        return self.file_raws
 
    def __getitem__(self,item):
        idx = self.index[0]
        data = self.samples[idx]
        self.index = self.index[1:]
        self.current_sample_num-=1
 
        if self.current_sample_num<=0:
        # all the samples in the memory have been used, need to get the new samples
            for _ in range(self.nraws):
                data = self.finput.readline()   # data contains the feature and label
                if data:
                    self.samples.append(data)
                else:
                    break
            self.current_sample_num = len(self.samples)
            self.index = list(range(self.current_sample_num))
            if self.shuffle:
                random.shuffle(self.samples)
 
        return data




if __name__=="__main__":
    datapath = r"inputdata\drastic.txt"
    batch_size = 64
    nraws = 1000
    epoch = 3
    train_dataset = Dataset(datapath,nraws)
    train_iter = Data.DataLoader(dataset = train_dataset, batch_size = batch_size)
    print(type(train_iter))
    '''
    for _ in range(epoch):
        train_dataset.initial()
        train_iter = Data.DataLoader(dataset = train_dataset, batch_size = batch_size)
        for _,data in enumerate(train_iter):
            print(type(data))
    '''