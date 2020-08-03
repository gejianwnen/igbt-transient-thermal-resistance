# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:44:27 2020

@author: admin
"""

import numpy as np
import torch
from torch import nn

from torch.utils.data import Dataset

class MyDatasetNASA(Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform
        self.size = x_train.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        traindata = self.x_train[idx]   # use skitimage
        label = self.y_train[idx]

        sample = {'traindata': traindata, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
# 
class MyDatasetZT(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        fh = open(df, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = pd.read_csv(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)
    
# a
        
    
        
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    