import math
import torch
from torch import nn
import numpy as np

cuda = torch.device('cuda')
def make_anchor(scale,img_w,img_h,ratios=(0.5,1,2)):
        
    
    sub_sample = scale/8
    
    w_size = int(img_w//sub_sample)
    h_size = int(img_h//sub_sample)
    
    ctr_x = torch.arange(sub_sample/2, (w_size+1) * sub_sample-(sub_sample/2), sub_sample)
    ctr_y = torch.arange(sub_sample/2, (h_size+1) * sub_sample-(sub_sample/2), sub_sample)
    
    ctr = torch.zeros((len(ctr_x)*len(ctr_y)),2)
    index = 0
    for y in range(len(ctr_y)):
        for x in range(len(ctr_x)):
            ctr[index, 0] = ctr_x[x]
            ctr[index, 1] = ctr_y[y] 
            index +=1
    
    anchors = torch.zeros((w_size * h_size * 3), 4)
    index = 0
    area = scale**2.0
    for c in ctr:
        ctr_x, ctr_y = c
        for i in range(len(ratios)):
            w = math.sqrt(area/ratios[i])
            h = ratios[i] * w
            anchors[index, 0] = ctr_x - w / 2.
            anchors[index, 1] = ctr_y - h / 2.
            anchors[index, 2] = ctr_x + w / 2.
            anchors[index, 3] = ctr_y + h / 2.
            index += 1
    
    return anchors


def anchorbox_generate(image=None,scales=(32,64,128,256,512),ratios=(0.5,1,2)):  #anchor를 5개 종류로 만들기
    img_w = image.shape[-1]
    img_h = image.shape[-2]
    anchor2 = make_anchor(scales[0],img_w,img_h).cuda()
    anchor3 = make_anchor(scales[1],img_w,img_h).cuda()
    anchor4 = make_anchor(scales[2],img_w,img_h).cuda()
    anchor5 = make_anchor(scales[3],img_w,img_h).cuda()
    anchor6 = make_anchor(scales[4],img_w,img_h).cuda()
    return [anchor2,anchor3,anchor4,anchor5,anchor6]



def labeling(image,GTboxes,sampling=False,sample_num=0):
    try:
        image = image.cuda()
        
        GTboxes = GTboxes.cuda()
        anchors = anchorbox_generate(image)
        
        if sampling == True:
            anc = anchors[sample_num]
        else:
            for tmp in range(len(anchors)):
                if tmp==0:
                    anc = anchors[0]
                else:
                    anc = torch.cat([anc,anchors[tmp]],dim=0)
        
        h_size = image.shape[-2]
        w_size=  image.shape[-1]
        #for k in range(1):
        anchor = anc
        index_inside = torch.where(
                (anchor[:, 0] >= 0) &
                (anchor[:, 1] >= 0) &
                (anchor[:, 2] <= w_size) &
                (anchor[:, 3] <= h_size)
            )[0].cuda()
        label = torch.full((len(index_inside), ),-1, dtype=torch.int32).cuda()  #int 타입으로 -1을 다 넣어주기
        
        valid_anchors = anchor[index_inside].cuda()    #anchor 중에 valid한 부분만 따로 valid로
        ious = torch.full((len(valid_anchors), GTboxes.shape[0]),0, dtype=torch.float32).cuda()  #ious는 일단 0으로 초기화
        
        xa1, ya1, xa2, ya2 = valid_anchors[:,0],valid_anchors[:,1],valid_anchors[:,2],valid_anchors[:,3]
        area = (xa2 - xa1) * (ya2 - ya1)
        for j in range(len(GTboxes)):
            xb1, yb1, xb2, yb2 = GTboxes[j][0],GTboxes[j][1],GTboxes[j][2],GTboxes[j][3]
            box_area =  (xb2 - xb1) * (yb2- yb1)
            x1 = torch.max(xb1, xa1)
            y1 = torch.max(yb1, ya1)
            x2 = torch.min(xb2, xa2)
            y2 = torch.min(yb2, ya2)
            share_area1 = torch.where(y2>y1, (y2 - y1) ,torch.tensor(0,dtype = torch.float).cuda())  
            share_area2 = torch.where(x2>x1, (x2 - x1) ,torch.tensor(0,dtype = torch.float).cuda())  
            share_area = share_area1 * share_area2 
            iou = share_area / (area + box_area - share_area)
            ious[:,j] = iou
        if len(ious)!=0: #ious가 없는 = valid anchor가 안 만들어지는 size의 이미지가 있을 수도 있다.
            gt_argmax_ious = ious.argmax(axis=0)
            gt_max_ious = ious[gt_argmax_ious, torch.arange(ious.shape[1])]
            
            argmax_ious = ious.argmax(axis=1)
            max_ious = ious[torch.arange(len(index_inside)).cuda(), argmax_ious.cuda()]
            gt_argmax_ious = torch.where(ious == gt_max_ious)[0].cuda()
            
            label[max_ious < 0.3] = 0
            label[gt_argmax_ious] = 1
            label[max_ious >= 0.7] = 1
            ###
            
            n_sample = 32
            
            pos_index = torch.where(label == 1)[0]
            
            if len(pos_index) > 16:        
                perm = torch.randperm(len(pos_index))
                idx = perm[:int(len(pos_index)-16)]
                disable_index = pos_index[idx]
                label[disable_index] = -1
            
            n_neg = n_sample - len(torch.where(label==1)[0])
            neg_index = torch.where(label == 0)[0]
            
            if len(neg_index) > n_neg:
                perm = torch.randperm(len(neg_index))
                idx = perm[:int(len(neg_index)-n_neg)]
                disable_index = neg_index[idx]
                label[disable_index] = -1
            
            #bbox labeling을 smooth loss 사용할 수 있게 만들자 
            max_bbox = GTboxes[argmax_ious]
            #anchor에서
            width = valid_anchors[:, 2] - valid_anchors[:, 0]
            height = valid_anchors[:, 3] - valid_anchors[:, 1]
            ctr_x = valid_anchors[:, 0] + 0.5 * width 
            ctr_y = valid_anchors[:, 1] + 0.5 * height
            #GT에서
            bbox_width = max_bbox[:, 2] - max_bbox[:, 0]
            bbox_height = max_bbox[:, 3] - max_bbox[:, 1]
            bbox_ctr_x = max_bbox[:, 0] + 0.5 * bbox_width
            bbox_ctr_y = max_bbox[:, 1] + 0.5 * bbox_height
            
            eps = torch.tensor(torch.finfo(height.dtype).eps).cuda()
            height = torch.maximum(height, eps).cuda()
            width = torch.maximum(width, eps).cuda()
            dy = (bbox_ctr_y - ctr_y) / height
            dx = (bbox_ctr_x - ctr_x) / width
            dh = torch.log(bbox_height / height).cuda()
            dw = torch.log(bbox_width / width).cuda()
            
            anchor_bbox = torch.vstack((dy, dx, dh, dw)).permute(1,0).cuda()
            
            anchor_labels = torch.full((len(anchor),), -1, dtype=label.dtype).cuda()
            anchor_labels[index_inside] = label

            anchor_locations = torch.full((len(anchor),4), 0, dtype=anchor_bbox.dtype).cuda()
            anchor_locations[index_inside, :] = anchor_bbox
            
        
        return anchor_labels,anchor_locations,anc
    except UnboundLocalError:
        print("no anchor")
        return None,None,None
    except RuntimeError:
        print("no anchor")
        return None,None,None