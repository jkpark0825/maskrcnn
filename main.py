from data import dataloader
from backbone  import model
from backbone  import resnet
from rpn import rpn
from rpn import trainer
from torchvision.ops import roi_align 
import torch
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from ROI_align import roi_align
from fasterRCNN import headtrainer
if __name__ =='__main__':
    directory = '../../Research/dataset/cocodataset'
    subname = 'val2017'
    
    data_loader = dataloader(directory,subname)
    
    for imgs, annotations in data_loader:
        a = imgs[0]
        print(a.shape)
        print(annotations[0]['bbox'].shape)
        print(annotations[0]['image_id'])
        break
    img=torch.stack([a],0)
    
    #trainer.train(data_loader)
    head = headtrainer.Headtrainer()
    head.train_GT(data_loader)
    
    #test
    #bbox,featuremap,scale = rpn.test(img) #bbox size = tensor(k,4)
    #aligned = roi_align(featuremap,[bbox],(7,7),scale,4)
    #이제 aligned을 faster rcnn에 넣자
