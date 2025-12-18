# Dataloader

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split

def loadData(filedir, batch_size=8):
    imagenet_means = (0.485, 0.456, 0.406)
    imagenet_stds = (0.229, 0.224, 0.225)


    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_means, imagenet_stds)])

    
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_means, imagenet_stds)])
    

    trainval_dataset = torchvision.datasets.ImageFolder(filedir)

    # split
    train_portion = 0.9


    all_idxes = np.arange(len(trainval_dataset))
    all_targets = trainval_dataset.targets

    train_idx, val_idx = train_test_split(all_idxes, train_size=train_portion, stratify = all_targets, random_state = 0)


    train_dataset = torch.utils.data.Subset(
    torchvision.datasets.ImageFolder(filedir, transform=train_transform),
    train_idx
)

    val_dataset = torch.utils.data.Subset(
    torchvision.datasets.ImageFolder(filedir, transform=val_transform),
    val_idx
)
    

    #Create dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataset, val_dataset, train_dataloader, val_dataloader
