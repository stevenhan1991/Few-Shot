import numpy as np
import torch
from skimage import data,img_as_float
from skimage.transform import rescale, resize
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from utils import *
from tqdm import tqdm

class VolumeDataSet(Dataset):
    def __init__(self,configs):
        self.task = configs.task
        self.path = configs.data.path
        self.factor = configs.data.factor
        self.total = configs.data.num
        self.res = configs.data.res
        self.cropsize = configs.data.cropsize
        self.samples = configs.data.training_samples
        self.var1 = configs.data.var1
        if self.task == 'V2V':
            self.var2 = configs.data.var2
        self.croptimes = configs.data.croptimes
        self.dataset = configs.data.dataset
        self.interval = (self.factor - 1)//2
        self.ReadData()

    def ReadData(self):
        obs = []
        gt = []
        
        for k in self.samples:
            print(k)
            if self.task == 'SSR':
                data = np.fromfile(self.path+self.var1+'/{:04d}.dat'.format(k),dtype='<f')
                data = data.reshape((self.res[2],self.res[1],self.res[0])).transpose()
                data = 2*(data-data.min())/(data.max()-data.min())-1

                data_ = resize(data,(self.res[0]//self.factor,self.res[1]//self.factor,self.res[2]//self.factor),order=3)

                data_ = resize(data_,(self.res[0],self.res[1],self.res[2]),order=3)
                data_ = np.clip(data_,-1.0,1.0)
                
            elif self.task == 'V2V':
                data_ = np.fromfile(self.path+self.var1+'/{:04d}.dat'.format(k),dtype='<f')
                data_ = data_.reshape((self.res[2],self.res[1],self.res[0])).transpose()
                data_ = 2.0 * (data_ - data_.min()) / (data_.max() - data_.min()) - 1.0

                data = np.fromfile(self.path+self.var2+'/{:04d}.dat'.format(k),dtype='<f')
                data = data.reshape((self.res[2],self.res[1],self.res[0])).transpose()
                data = 2.0 * (data - data.min())/(data.max() - data.min()) - 1.0
                if self.var2 in ['H+','PD']:
                    data = - data

            elif self.task == 'TSR':
                
                d = []
                data = np.fromfile(self.path+self.var1+'/{:04d}.dat'.format(k),dtype='<f')
                data = data.reshape((self.res[2],self.res[1],self.res[0])).transpose()
                data = 2*(data-data.min())/(data.max()-data.min())-1

                for i in range(self.interval,0,-1):
                    #print(k-i)
                    data_ = np.fromfile(self.path+self.var1+'/{:04d}.dat'.format(k-i),dtype='<f')
                    data_ = data_.reshape((self.res[2],self.res[1],self.res[0])).transpose()
                    data_ = 2*(data_-data_.min())/(data_.max()-data_.min())-1
                    d.append(data_)

                d.append(data)

                for i in range(1,self.interval+1):
                    #print(k+i)
                    data_ = np.fromfile(self.path+self.var1+'/{:04d}.dat'.format(k+i),dtype='<f')
                    data_ = data_.reshape((self.res[2],self.res[1],self.res[0])).transpose()
                    data_ = 2*(data_-data_.min())/(data_.max()-data_.min())-1
                    d.append(data_)

                data = np.stack(d)

                d = []

                data_ = np.fromfile(self.path+self.var1+'/{:04d}.dat'.format(k-self.interval-1),dtype='<f')
                data_ = data_.reshape((self.res[2],self.res[1],self.res[0])).transpose()
                data_ = 2*(data_-data_.min())/(data_.max()-data_.min())-1
                d.append(data_)


                data_ = np.fromfile(self.path+self.var1+'/{:04d}.dat'.format(k+self.interval+1),dtype='<f')
                data_ = data_.reshape((self.res[2],self.res[1],self.res[0])).transpose()
                data_ = 2*(data_-data_.min())/(data_.max()-data_.min())-1
                d.append(data_)

                data_ = np.stack(d)

            obs.append(data_)
            gt.append(data)

        self.obs = np.asarray(obs)
        self.gt = np.asarray(gt)

        print(len(gt))

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx // self.croptimes


        if self.res[0] == self.cropsize[0]:
            x = 0
        else:
            if self.dataset in ['Ionization']:
                x = np.random.randint(280,self.res[0]-self.cropsize[0])
            else:
                x = np.random.randint(0,self.res[0]-self.cropsize[0])

        if self.res[1] == self.cropsize[1]:
            y = 0
        else:
            if self.dataset in ['Combustion']:
                y = np.random.randint(118,622-self.cropsize[1])
            else:
                y = np.random.randint(0,self.res[1]-self.cropsize[1])

        if self.res[2] == self.cropsize[2]:
            z = 0
        else:
            z = np.random.randint(0,self.res[2]-self.cropsize[2])

        if self.task == 'SSR':
            obs = self.obs[idx][x:(x+self.cropsize[0]),y:(y+self.cropsize[1]),z:(z+self.cropsize[2])]
            gt = self.gt[idx][x:(x+self.cropsize[0]),y:(y+self.cropsize[1]),z:(z+self.cropsize[2])]
        elif self.task == 'V2V':
            obs = self.obs[idx][x:(x+self.cropsize[0]),y:(y+self.cropsize[1]),z:(z+self.cropsize[2])]
            gt = self.gt[idx][x:(x+self.cropsize[0]),y:(y+self.cropsize[1]),z:(z+self.cropsize[2])]
        elif self.task == 'TSR':
            obs = self.obs[idx][:,x:(x+self.cropsize[0]),y:(y+self.cropsize[1]),z:(z+self.cropsize[2])]
            gt = self.gt[idx][:,x:(x+self.cropsize[0]),y:(y+self.cropsize[1]),z:(z+self.cropsize[2])]
        return {'input':obs, 'output':gt}

    def __len__(self):
        return self.croptimes * len(self.gt)
