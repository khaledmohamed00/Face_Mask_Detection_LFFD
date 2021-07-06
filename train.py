
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import BasketNet
from torchvision import transforms
from dataset import FacemaskDataset
from lossfn import FacemaskLoss
from tqdm import tqdm
from augmentor import FacemaskAug
from config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    transform = FacemaskAug()
    dataset = FacemaskDataset("/content/face_mask_detection",transform = transform)
    dataloader = DataLoader(dataset,batch_size = 2,shuffle =  True,num_workers = 2,pin_memory=True)
    
    net = FacemaskNet(num_classes = num_classes)
    net.to(device)
    optimizer = optim.Adam(net.parameters(),lr = 1e-3)    
    loss_fn = FacemaskLoss()
    num_batches = len(dataloader)
    min_loss = float("inf")
    for epoch in range(num_epochs):
        epoch_loss_cls = 0.
        epoch_loss_reg = 0.

        for img,gt_pos,gt_labels,not_ignored in tqdm(dataloader):
            img = img.to(device)
            gt_pos =gt_pos.to(device)
            not_ignored = not_ignored.to(device)
            gt_labels =gt_labels.to(device)
            cls,loc = net(img)
            reg_loss,cls_loss = loss_fn(cls,loc,gt_labels,gt_pos,not_ignored)
            epoch_loss_cls += cls_loss.item()
            epoch_loss_reg += reg_loss.item()
            loss = (reg_loss +  cls_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("cls_loss:{}---reg_loss:{}".format(epoch_loss_cls/num_batches,epoch_loss_reg/num_batches))
        if (epoch_loss_cls/num_batches + epoch_loss_reg/num_batches) < min_loss:
            min_loss = epoch_loss_cls/num_batches + epoch_loss_reg/num_batches
            torch.save(net.state_dict(),"./ckpt/{}.pth".format(int(epoch_loss_cls/num_batches * 1000 + epoch_loss_reg/num_batches * 1000)))