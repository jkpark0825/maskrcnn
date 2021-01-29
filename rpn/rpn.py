import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from backbone import model

class rpn(nn.Module):
    def __init__(self):
        super(rpn,self).__init__()
        self.conv1 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(256,3*2,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(256,3*4,kernel_size=1,stride=1,padding=0)
        
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()

        self.conv2.weight.data.normal_(0, 0.01)
        self.conv2.bias.data.zero_()

        self.conv3.weight.data.normal_(0, 0.01)
        self.conv3.bias.data.zero_()

        
        self.softmax = nn.Softmax(dim=1)
        ##
    def forward(self,x):
       
        x = F.relu(self.conv1(x))
        classify = self.conv2(x)
        bbox = self.conv3(x)
        
        classify = classify.permute(0,2,3,1).contiguous()
        ###
        classify  = classify.view(1,-1,2)[0]
        classify = self.softmax(classify) #해야하나?
        ###
        bbox = bbox.permute(0,2,3,1).contiguous()
        bbox = bbox.view(1,-1,4)[0]


        return classify,bbox

    


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fpn = model.FPN()
        self.resnets = model.resnet()
        self.rpnmodel2 = rpn()
        self.rpnmodel3 = rpn()
        self.rpnmodel4 = rpn()
        self.rpnmodel5 = rpn()
        self.rpnmodel6 = rpn()
        
    def forward(self,x):
        try:
            p2, p3, p4, p5, p6 = self.fpn(x)
            
            pred_c2, pred_b2=self.rpnmodel2(p2)
            pred_c3, pred_b3=self.rpnmodel3(p3)
            pred_c4, pred_b4=self.rpnmodel4(p4)
            pred_c5, pred_b5=self.rpnmodel5(p5)
            pred_c6, pred_b6=self.rpnmodel6(p6)
            pred_c = torch.cat([pred_c2,pred_c3,pred_c4,pred_c5,pred_c6],dim=0)
            pred_b = torch.cat([pred_b2,pred_b3,pred_b4,pred_b5,pred_b6],dim=0)
            

            return pred_c, pred_b,p2, p3, p4, p5, p6

        except RuntimeError:
            print("input size error")
            return None,None,None,None,None,None,None

