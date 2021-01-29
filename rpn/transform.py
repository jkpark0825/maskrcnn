import os
import numpy as np
import torch

def bbox_transform(bbox=None, anchor = None): #cls가 1로 나오는 bbox들을 집어넣는다. shape = (n,4), anchor는 anchor 종류0~4->scale 결정, anchor_num몇번인지     
    if bbox.dim()==1:
        width = anchor[2]-anchor[0]
        height = anchor[3]-anchor[1]
        ctr_x = anchor[0]+0.5*width
        ctr_y = anchor[1]+0.5*height

        dy,dx,dh,dw = bbox[0],bbox[1],bbox[2],bbox[3]
        
        bbox_ctr_x = width *dx + ctr_x
        bbox_ctr_y = height *dy + ctr_y
        bbox_width = torch.exp(dw) * width
        bbox_height = torch.exp(dh) * height

        bbox_x1 = bbox_ctr_x - 0.5 * bbox_width
        bbox_y1 =  bbox_ctr_y - 0.5 * bbox_height
        bbox_x2 = bbox_ctr_x + 0.5 * bbox_width
        bbox_y2 =  bbox_ctr_y + 0.5 * bbox_height
        transformed = torch.stack([bbox_x1,bbox_y1,bbox_x2,bbox_y2],dim=1)

    else:
        width = anchor[:,2]-anchor[:,0]
        height = anchor[:,3]-anchor[:,1]
        ctr_x = anchor[:,0]+0.5*width
        ctr_y = anchor[:,1]+0.5*height

        dy,dx,dh,dw = bbox[:,0],bbox[:,1],bbox[:,2],bbox[:,3]
        
        bbox_ctr_x = width *dx + ctr_x
        bbox_ctr_y = height *dy + ctr_y
        bbox_width = torch.exp(dw) * width
        bbox_height = torch.exp(dh) * height

        bbox_x1 = bbox_ctr_x - 0.5 * bbox_width
        bbox_y1 =  bbox_ctr_y - 0.5 * bbox_height
        bbox_x2 = bbox_ctr_x + 0.5 * bbox_width
        bbox_y2 =  bbox_ctr_y + 0.5 * bbox_height
        
        
        transformed = torch.stack([bbox_x1,bbox_y1,bbox_x2,bbox_y2],dim=1)
    
    return transformed