import os
import numpy as np
import torch
import random
from torchvision.ops import roi_align 
from . import head
from rpn import rpn
from rpn import trainer
from rpn import transform
import torch.nn.functional as F
import torch.optim as optim
from fvcore.nn import smooth_l1_loss
from rpn import trainer
from rpn import testanchor
from rpn import visualizeimage

class Headtrainer():
    def __init__(self):
        super(Headtrainer,self).__init__()
        net2 = rpn.Net()
        net2 = net2.cuda()
        optimizer2 = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)
        net2,self.optimizer2 = trainer.load(optimizer2,'rpn/model_final_1/')
        self.net2 = net2.cuda()
        
        net = head.head()
        self.net = net.cuda()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        ###
        '''
        net,self.optimizer = self.load(self.optimizer,'./fasterRCNN/model/')
        self.net = net.cuda()
        '''
        ###
        
    def save(self,model=None, optimizer=None,path=None):
        print('===> saving model')
        PATH = path
        torch.save(model, PATH + 'model.pt')  
        torch.save(model.state_dict(), PATH + 'model_state_dict.pt')
        torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, PATH + 'all.tar') 
        
        return

    def load(self,optimizer = None,path=None):
        print('===> loading model')
        PATH = path
        model = torch.load(PATH + 'model.pt')  
        model.load_state_dict(torch.load(PATH + 'model_state_dict.pt')) 
        checkpoint = torch.load(PATH + 'all.tar')  
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer

    def train_GT(self,dataloader):
        for epoch in range(1000):
            for i,data in enumerate(dataloader):
                imgs, annotations = data
                img = imgs[0].clone()
                img = torch.stack([img],0).cuda() # 1 3 H W 
                bbox = annotations[0]['bbox']
                
                
                pred_cs,pred_bs, p2, p3, p4, p5, p6 = self.net2(img)
                if pred_cs !=None:
                    featuremap = [p2,p3,p4,p5,p6]
                    beforelength=0
                    trans_final_bbox = None
                    score=None
                    for anchornum in range(5):
                        
                        GT_c, GT_b, anchors = testanchor.labeling(img,annotations[0]['bbox'],sampling=True,sample_num=anchornum)#0,1,2,3,4
                        roi_aligned=torch.zeros((1))
                        if GT_c!=None:
                            pred_b = pred_bs[beforelength:beforelength+len(GT_b)]
                            pred_c = pred_cs[beforelength:beforelength+len(GT_c)]
                            transformed = transform.bbox_transform(pred_b,anchors)
                            
                            beforelength = beforelength+len(GT_b)
                            index_true = torch.where(pred_c[:,1] > 0.7)[0].cuda()
                            index_true = torch.where(GT_c ==1)[0].cuda()
                            index_false = torch.where(pred_c[:,1] < 0.3)[0].cuda()
                            
                            if len(index_true) > 100:        
                                perm = torch.randperm(len(index_true))
                                idx = perm[:100]
                                index_true = index_true[idx]
                            if len(index_false)> len(index_true):
                                perm = torch.randperm(len(index_false))
                                idx = perm[:len(index_true)]
                                index_false = index_false[idx]
                            
                            
                            
                            indexs = torch.cat([index_true,index_false],dim=0)
                            
                            if len(index_true)!=0:  
                                for num,k in enumerate(indexs,0):
                                    if num==0:
                                        roi_aligned = roi_align(featuremap[anchornum],[torch.stack([transformed[k]],dim=0)],(7,7),1/(2**(anchornum+2)),4,True)
                                    else:
                                        roi_aligned = torch.cat([roi_aligned,roi_align(featuremap[anchornum],[torch.stack([transformed[k]],dim=0)],(7,7),1/(2**(anchornum+2)),4,True)],dim=0) # 7 x 7 256 
                                    
                            ###
                            #for tempepoch in range(1000):
                            if len(roi_aligned)!=1:
                                label = torch.cat([torch.ones((int(len(index_true)),)),torch.zeros((int(len(index_false)),))],dim=0)
                                self.optimizer.zero_grad()
                                pred_classify, pred_bbox = self.net(roi_aligned)
                                loss_c = F.cross_entropy(pred_classify.cuda(), label.long().cuda())
                                loss_b = smooth_l1_loss(pred_bbox[:len(index_true)].cuda(),GT_b[index_true].cuda(),1.0,reduction='sum')/len(index_true)
                                loss=loss_c+loss_b
                                print('step >> ',end='')
                                print(epoch, i, end='')
                                print('lossTotal >>',end='')
                                print(loss.item(),end='')
                                print('lossC >>',end='')
                                print(loss_c.item(),end='')
                                print('lossB >>',end='')
                                print(loss_b.item())
                                loss.backward(retain_graph=True)
                                self.optimizer.step()

                                
                                anc = anchors[index_true].cuda()
                                
                                if anchornum==0:
                                    bbox_index = torch.where(pred_classify[:len(index_true)][:,1] > 0.7)[0].cuda()
                                    finalbbox = pred_bbox[bbox_index]
                                    if len(finalbbox)!=0:
                                        trans_final_bbox = transform.bbox_transform(finalbbox,anc[bbox_index])
                                        score = pred_classify[bbox_index].cuda()
                                else:
                                    if trans_final_bbox==None:
                                        bbox_index = torch.where(pred_classify[:len(index_true)][:,1] > 0.7)[0].cuda()
                                        finalbbox = pred_bbox[bbox_index]
                                        if len(finalbbox)!=0:
                                            trans_final_bbox = transform.bbox_transform(finalbbox,anc[bbox_index])
                                            score = pred_classify[bbox_index].cuda()
                                    else:
                                        bbox_index = torch.where(pred_classify[:len(index_true)][:,1] > 0.7)[0].cuda()
                                        finalbbox = pred_bbox[bbox_index]
                                        if len(finalbbox)!=0:
                                            trans_final_bbox = torch.cat([trans_final_bbox,transform.bbox_transform(finalbbox,anc[bbox_index])],dim=0)
                                            score = torch.cat([score,pred_classify[bbox_index]],dim=0).cuda()
                                            
                                if (i+1)%100==0 and anchornum==3 or anchornum==4:
                                    visualizeimage.Visualize(img,trans_final_bbox,score,path = 'final_result'+str(anchornum)+'.png',is_nms=True)
                                    self.save(self.net,self.optimizer,'./fasterRCNN/model/')
                                    
                    



