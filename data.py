import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import torch.nn.functional as F
import numpy as np
class cocodataset(Dataset):
    def __init__(self, directory, subname, transforms=None):
        self.directory = directory
        self.transforms = transforms
        self.subname = subname
        path = os.path.join(self.directory,'annotations','instances_'+subname+'.json')
        self.coco = COCO(path)
        whole_img_id = self.coco.getImgIds() 
        self.personimg_id = []
        person_id = self.coco.getCatIds(catNms=['person']) #probably 1
        self.personimg_id = self.coco.getImgIds(catIds = person_id) #person including id
        self.ids = list(sorted(self.coco.imgs.keys()))
    def __getitem__(self, index):
        #self.coco.loadImgs(self.image_ids[image_index])[0]
        coco = self.coco #whole
        img_id = self.personimg_id[index] #one image id
        
        ann_ids = coco.getAnnIds(imgIds = img_id)#annts per a image
        coco_annotations = coco.loadAnns(ann_ids) #annot read can be many cats
        person_annot = []
        for i in range(len(coco_annotations)):
            if coco_annotations[i]['category_id']==1: #only person
                person_annot.append(coco_annotations[i])
        boxes = np.zeros((0,4))
        for num, dictionary in enumerate(person_annot):
            box = np.zeros((1,4))
            box[0,:4] = dictionary['bbox']
            boxes = np.append(boxes,box,axis=0) #boxes = [x,y,w,h]
    
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]
        boxes = torch.as_tensor(boxes,dtype = torch.float32)
        labels = torch.as_tensor(torch.ones((boxes.shape[0],1)),dtype=torch.int64)
        #boxes and labels per image_id

        #print(coco.loadImgs(img_id)[0]['file_name'])
        filename = coco.loadImgs(img_id)[0]['file_name']      
        img = Image.open(os.path.join(self.directory,'images',self.subname,filename))
        
        img = torchvision.transforms.ToTensor()(img)
        c1=0
        c2=0
        if img.shape[1]%32!=0:
            a1 = img.shape[1]//32 #몫
            b1 = img.shape[1]-a1*32
            c1 = 32-b1
        if img.shape[2]%32!=0:
            a2 = img.shape[2]//32 #몫
            b2 = img.shape[2]-a2*32
            c2 = 32-b2
        #c2 = 960-img.shape[2]
        #c1 = 960-img.shape[1]
        pad = (0,c2,0,c1)
        img=F.pad(img,pad=pad,value=0)
        annotation = {}
        annotation["bbox"] = boxes
        annotation["labels"] = labels
        annotation["image_id"] = img_id
        
        return img, annotation
        
    def __len__(self):
        return len(self.personimg_id)
    def collate_fn(self,batch):
        return tuple(zip(*batch))
        
def dataloader(directory,subname):
    dataset = cocodataset(directory= directory,
                              subname = subname
                              )
    data_loader = DataLoader(dataset,
                             batch_size = 2,
                             shuffle=True,
                             num_workers=4,
                             collate_fn = dataset.collate_fn)
    return data_loader 
'''
if __name__ =='__main__':
    directory = '../../Research/dataset/cocodataset'
    subname = 'val2017'
    
    testdataset = cocodataset(directory= directory,
                              subname = subname
                              )
    data_loader = DataLoader(testdataset,
                             batch_size = 1,
                             shuffle=True,
                             num_workers=4,
                             collate_fn = testdataset.collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    for imgs, annotations in data_loader:
        print((imgs[0].shape))
        print(annotations[0]['image_id'])
        
        break

        '''
        
        
