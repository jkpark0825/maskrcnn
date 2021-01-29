import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from . import nms


def Visualize(img = None, bbox=None,top=None,path=None,is_nms = True):
    try:
        img_h = img.shape[-2]
        img_w = img.shape[-1]
        img2 = img.clone().detach()
        ##################NMS여기다가
        if top!=None:
            score = top.cuda()
            score = score[:,1]
            
            ord = score.argsort(dim=0).cuda()
            if is_nms ==True:           
                bbox  = nms.NMS_algo(ord,bbox,img_w,img_h)
            print(bbox)
        for i in range(bbox.shape[0]):
            x1,y1,x2,y2 = bbox[i]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            if x2>img_w: x2 = img_w-1
            if y2>img_h :y2 = img_h-1
            if x1<0 : x1 = 0 
            if y1<0 : y1 = 0
            if x2 < img_w and x1>=0 and y1>=0 and y2 < img_h and x2>x1 and y2>y1:
                img2[0,:,y1,x1:x2] = 0
                img2[0,:,y2,x1:x2] = 0
                img2[0,:,y1:y2,x1] = 0
                img2[0,:,y1:y2,x2] = 0
        
        img = transforms.ToPILImage()(img[0]).convert("RGB")
        img2 = transforms.ToPILImage()(img2[0]).convert("RGB")
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(img2)
        
        plt.show()
        plt.savefig(path)
        return 
    except AttributeError:
        print("image error")
        return
    except RuntimeError:
        print("runtime error")
        return