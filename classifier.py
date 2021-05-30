'''
method 1 for task 5.
## implement a simple convolution network act as classifier using pytorch.
### contains 4 layers, each has conv2d,bn,activ layer.
'''

import torch
from torch import nn
import torch.nn.functional as F


class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        

        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.pool(output)
        output = self.bn(output)
        output = self.relu(output)

        return output
    
    
class Classifier(nn.Module):
    def __init__(self,layers,base_nc=16):
        super().__init__()
        self.in_channels=1
        self.base_nc=base_nc
        for i in range(layers):
            m = Unit(in_channels=self.in_channels,out_channels=self.base_nc)
            setattr(self,"conv{}".format(i),m)
            self.in_channels=self.base_nc
            self.base_nc = self.base_nc*2
        self.fc = nn.Linear(in_features=100*100*self.base_nc//2,out_features=2) #some question here. the dimension match has some glitch.
        self.sequence = nn.Sequential(*[getattr(self,"conv{}".format(i)) for i in range(layers)],self.fc) #problem:without flatten
        
    def forward(self,x):
        output = self.sequence(x)
        return output
    