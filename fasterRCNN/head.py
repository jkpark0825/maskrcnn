import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class head(nn.Module):
    def __init__(self):
        super(head,self).__init__()
        self.conv1 = nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
        
        #self.deconv1= nn.ConvTranspose2d(128, 128, 64, stride = 2, padding=0)
        
        self.conv2 = nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 32)
        self.fc3 = nn.Linear(32, 2)

        self.bb1 = nn.Linear(64*7*7,120)
        self.bb2 = nn.Linear(120,4)
        self.softmax = nn.Softmax(dim=1)
        ##
    def forward(self,x):
        #print(x[1][:,:,1])
        #print(x[-1][:,:,1])
        x = F.relu(self.conv1(x))
        #x = self.deconv1(x) 
        x = F.relu(self.conv2(x))
        x = x.view(-1,7*7*64)
        classify = F.relu(self.fc1(x))
        classify = F.relu(self.fc2(classify))
        classify = self.fc3(classify)
        classify = self.softmax(classify)
        bbox = self.bb1(x)
        bbox = self.bb2(bbox)
        return classify,bbox
    