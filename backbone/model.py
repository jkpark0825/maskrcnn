import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .resnet import ResNet
from .resnet import resblock
class resnet(nn.Module):
    def __init__(self):
        super(resnet,self).__init__()
        
        self.resnetblock1 = ResNet(resblock, [3, 4, 6, 3])
        
    def forward(self, x):
        
        c2,c3,c4,c5 = self.resnetblock1(x)
       
        return c2,c3,c4,c5

class topdown(nn.Module):
    def __init__(self,small_channels,big_channels, out_channels=256):
        super(topdown,self).__init__()
        self.conv1 = nn.Conv2d(small_channels, out_channels,kernel_size=1,stride=1)
        self.conv2 = nn.Conv2d(big_channels, out_channels,kernel_size=1,stride=1)
        self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1,padding=1)
    def forward(self,small,big):
        #small = F.upsample(self.conv1(small), scale_factor=2)
        temp = self.conv1(small)
        small = F.interpolate(temp,mode='nearest',scale_factor=2)
        big = self.conv2(big)
        out = (small+big)/2
        
        return self.conv3(out),self.conv3(temp)

def fpn(image):
    resnets = resnet()
    c2,c3,c4,c5 = resnets(image)
    fpn4 = topdown(2048,1024)
    fpn3 = topdown(1024,512)
    fpn2 = topdown(512,256)
    
     
    p4,p5 = fpn4(c5,c4)
    p3,_ = fpn3(c4,c3)
    p2,_ = fpn2(c3,c2)
    p6 = F.max_pool2d(p5,kernel_size=1,stride=2,padding=0)
    return p2,p3,p4,p5,p6

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.resnet = ResNet(resblock, [3, 4, 6, 3])
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c2,c3,c4,c5 = self.resnet(x)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        #p5 = self.smooth4(p5)
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        p6 = F.max_pool2d(p5,kernel_size=1,stride=2,padding=0)
        return p2, p3, p4, p5, p6
    


