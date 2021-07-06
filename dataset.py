import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np  
from priors import Priors
from Facemask_utils import *
np.set_printoptions(threshold=np.inf)  


class FacemaskDataset(Dataset):
    def __init__(self,img_path,transform = None):
        self.img_paths = glob.glob(img_path + "/images/*.png")
        self.labels = [label.replace(".png",".xml").replace("images","annotations") for label in self.img_paths]
        self.class_names = ("mask_weared_incorrect","without_mask","with_mask")
        prior = Priors() 
        self.priors = prior() # center form
        self.imgW,self.imgH = 640,640
        self.transform = transform  
    def __getitem__(self,idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label_file = self.labels[idx]
        gt_bboxes,gt_classes = self._get_annotation(idx)
        if self.transform:
            img,gt_bboxes,gt_classes = self.transform(np.array(img),gt_bboxes,gt_classes)

        gt_bboxes = torch.tensor(gt_bboxes)
        gt_classes = torch.LongTensor(gt_classes)
        gt_bboxes,gt_classes,ignored = assign_priors(gt_bboxes,gt_classes,self.priors,0.5) 
        locations = convert_boxes_to_locations(gt_bboxes, self.priors,2) 

        return [img,locations,gt_classes,ignored]
        #return img,locations,gt_classes
    
    def _get_annotation(self,idx):
        annotation_file = self.labels[idx]
        objects = ET.parse(annotation_file).findall("object")
        size = ET.parse(annotation_file).find("size")
        boxes = []
        labels = []
        #is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            imgW = float(size.find('width').text)
            imgH = float(size.find('height').text)
            boxes.append([x1/imgW,y1/imgH,x2/imgW,y2/imgH])
            labels.append(self.class_names.index(class_name))
        return boxes,labels
    def __len__(self):
        return len(self.img_paths)
if __name__ == '__main__':
    import random
    from augmentor import FacemasktAug
    transform = FacemaskAug(to_tensor = False)
    import cv2 as cv
    datset = FacemaskDataset("/content/face_mask_detection",transform = transform)
    img,gt_loc,gt_labels,ignored = datset[random.choice(range(len(datset)))]
    cv_img=img.permute(1,2,0).numpy()
    cv_img = cv.cvtColor(cv_img,cv.COLOR_RGB2BGR)  
    priors = datset.priors
    idx = (gt_labels > 0) & ignored.bool()
    loc = convert_locations_to_boxes(gt_loc,datset.priors,2)
    loc = loc[idx]
    priors = priors[idx]
    label = gt_labels[idx]
    '''
    for i in range(priors.size(0)):
        
        x,y,r = priors[i,:]
        #print(x,y,r)
        x = x.item() * 640
        y = y.item() * 640
        r = r.item() * 640
        print(x,y,r)
        cv.circle(cv_img,(int(x),int(y)),int(r),(255,0,0),2)
    '''
    for i in range(loc.shape[0]):
    #print(loc.shape)    
        x1,y1,x2,y2=loc[i,:]*640

        cv.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

    cv.imwrite('00.jpg',cv_img*255.)