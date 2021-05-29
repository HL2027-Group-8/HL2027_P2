import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
from random import choice
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor

class NiftiImage(Dataset):
    def __init__(self,labels,img_dir = "./data/Resized/COMMON/",transform = None):
        super().__init__()
        self.img_dir = img_dir
        self.img_labels = labels
        self.img_files = self.read_img(img_dir=img_dir)
        self.transform = transform
        
    def normalize(self, data, minmax):
        '''
        Apply the normalization to make the data between [-1,1]
        '''
        data_min, data_max = minmax
        data = np.clip(data, data_min, data_max)
        data = (data - data_min) / (data_max - data_min)
        data = data * 2.0 - 1.0
        return data 

    def to_tensor(self, data, minmax):
        '''
        normalize and to tensor.
        '''
        if data.ndim == 2: data = data[np.newaxis, ...]
        data = self.normalize(data, minmax)
        data = torch.FloatTensor(data)
        return data
    
    def read_img(self,img_dir):
        img_files = self.file_name(img_dir=img_dir)
        img_stack = []
        for img_file in img_files:
            img = sitk.ReadImage(path.join(img_dir,img_file))
            img  = sitk.GetArrayFromImage(img)
            #####
            img_stack += [img[i] for i in range(img.shape[0])]#num of slice
        return img_stack     
     
    def file_name(self,img_dir):
        L=[]   
        path_list = os.listdir(img_dir)
        path_list.sort()
        for filename in path_list:
            if 'nii' in filename and 'image' in filename:
                L.append(path.join(filename))   
        return L
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img = self.img_files[index]
        minmax = (np.min(img),np.max(img))
        data = self.to_tensor(img,minmax)
        if self.transform is not None:
            data = self.transform(data)        
        label = self.img_labels[index]
        return data,label

        
        