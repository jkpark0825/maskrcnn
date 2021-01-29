import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from backbone import model
from . import bbox_loss
from . import testanchor
from . import transform
from . import visualizeimage
from . import rpn
from fvcore.nn import smooth_l1_loss

def printpoint(epoch,i,loss,loss_c,loss_b):
    print("step>>",end='')
    print(str(epoch),str(i)+' ',end='')
    
    print("loss>>",end='')
    print(str(loss.item())+' ',end='')
    print("loss_c>>",end='')
    print(str(loss_c.item())+' ',end='')
    print("loss_b>>",end='')    
    print(loss_b.item())
    return
def save(model=None, optimizer=None,path=None):
    print('===> saving model')
    PATH = path
    torch.save(model, PATH + 'model.pt')  
    torch.save(model.state_dict(), PATH + 'model_state_dict.pt')
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + 'all.tar') 
    
    return

def load(optimizer = None,path=None):
    print('===> loading model')
    PATH = path
    model = torch.load(PATH + 'model.pt')  
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt')) 
    checkpoint = torch.load(PATH + 'all.tar')  
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer



def train(dataloader):
    print('===> Initializing trainer')
    cuda = torch.device('cuda')
    net = rpn.Net()
    net = net.cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss=0.0
    net,optimizer = load(optimizer,'./rpn/model_final_1/')
    net = net.cuda()
    for epoch in range(1000):    
        for i,data in enumerate(dataloader):
            imgs, annotations =data
            img = imgs[0]
            img = torch.stack([img],0).cuda() # 1 3 H W 
            GT_c, GT_b, anchors = testanchor.labeling(img,annotations[0]['bbox'])
            for j in range(10):
                optimizer.zero_grad()
                pred_c,pred_b,_,_,_,_,_ = net(img)
                if pred_c != None:
                    GT_c = GT_c.to(torch.float32)
                    pos_id = torch.where(GT_c ==1)[0].cuda()
                    neg_id = torch.where(GT_c ==0)[0].cuda()


                    '''
                    if len(pos_id) > 16:        
                        perm = torch.randperm(len(pos_id))
                        idx = perm[:int(len(pos_id)-16)]
                        pos_id = pos_id[idx]
                    
                    if len(neg_id) > 32-len(pos_id):
                        perm = torch.randperm(len(neg_id))
                        idx = perm[:int(len(neg_id)-n_neg)]
                        neg_id = neg_id[idx]
                    '''
                    if len(pos_id) > 16:        
                        perm = torch.randperm(len(pos_id))
                        idx = perm[:int(len(pos_id)-16)]
                        pos_id = pos_id[idx]
                    
                    if len(neg_id) > len(pos_id):
                        perm = torch.randperm(len(neg_id))
                        idx = perm[:len(pos_id)]
                        neg_id = neg_id[idx]
                     
                    val_id = pos_id.clone()
                    val_id = torch.cat([val_id,neg_id],dim=0)

                    loss_c = F.cross_entropy((pred_c[val_id]).cuda(), GT_c[val_id].long().cuda(),reduction ='sum') / len(val_id)
                    
                    #loss_b = bbox_loss(pred_b[pos_id].cuda(),GT_b[pos_id].cuda(),1) 
                    loss_b = smooth_l1_loss(pred_b[pos_id].cuda(),GT_b[pos_id].cuda(),1.0,reduction ='sum') / (len(val_id)) 

                    loss =  loss_c+loss_b
                    
                    loss.backward()
                    optimizer.step()
                    
                    printpoint(epoch,i,loss,loss_c,loss_b)
                    
                    if (i+1) % 2 == 0 and j==9:
                        index = torch.where(pred_c[:,1] > 0.9)[0].cuda()
                        index = torch.where(GT_c ==1)[0].cuda()
                        
                        score = pred_c[index]    
                        print(pred_b[index].device)
                        print(anchors[index].device)
                        print('===> saving middle image')
                        transformed = transform.bbox_transform(pred_b[index],anchors[index])
                        
                        visualizeimage.Visualize(img,transformed,score,path = 'predicted_final_1.png',is_nms=False)
                        visualizeimage.Visualize(img,transformed,score,path = 'predicted_final_NMS_1.png',is_nms=True)

                        transformed = transform.bbox_transform(GT_b[index],anchors[index])
                        visualizeimage.Visualize(img,transformed,score,path = 'GT_final_1.png',is_nms=True)
                        
                    if (i+1) % 200 == 0 and j==9:
                        save(net,optimizer,'./rpn/model_final_1/')
    #모델 파일명들이랑