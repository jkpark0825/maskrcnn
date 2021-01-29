import os
import numpy as np
import torch
def box_area1(box):
    return(box[:,2]-box[:,0])*(box[:,3]-box[:,1]).cuda()
def box_area2(box):
    return(box[2]-box[0])*(box[3]-box[1]).cuda()

def iou(box1,box2):
    x0 = torch.max(box1[:,0],box2[0]).cuda()
    x1 = torch.min(box1[:,2],box2[2]).cuda()
    y0 = torch.max(box1[:,1],box2[1]).cuda()
    y1 = torch.min(box1[:,3],box2[3]).cuda()
    
    share_area1 = torch.where(y1>y0, (y1 - y0) ,torch.tensor(0,dtype = torch.float).cuda())  
    share_area2 = torch.where(x1>x0, (x1 - x0) ,torch.tensor(0,dtype = torch.float).cuda())  
            
    area = share_area1*share_area2.cuda()
    
    return (area / (box_area1(box1)+box_area2(box2)-area))

def NMS_algo(ord,bbox,img_w,img_h):
    keep = torch.ones(len(ord)).cuda()
    if len(ord)>300:
        order = ord[int(len(ord)-300):].cuda()
        keep[ord[:int(len(ord)-300)]] = 0
    else:
        order = ord
    bbox[:,0:4:2] = torch.clip(bbox[:,0:4:2],0,img_w)
    bbox[:,1:4:2] = torch.clip(bbox[:,1:4:2],0,img_h)

    for t in range(len(order)-1):
        ovps = iou(bbox[order[t+1:]],bbox[order[t]])
        for k,ov in enumerate(ovps):
            if ov>0.5:
                keep[order[k+t+1]]=0
    interest = torch.where(keep==1)[0].cuda()
    

    bbox = bbox[interest]

    return bbox