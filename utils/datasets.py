import os
import torch
import torchvision

from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_imagenet(datadir, train=True, bs=256):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    get_dataset = getattr(datasets, "ImageNet")

    normalize = torchvision.transforms.Normalize(mean=mean, std=std)
    tr_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), normalize])
    val_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), normalize])
    if train is True:
        dataset = get_dataset(root=datadir, split='train', transform=tr_transform)
    else:
        dataset = get_dataset(root=datadir, split='val', transform=val_transform)

    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)

    return data_loader


def get_cifar10(train=True, bs=512): #8
    path   = os.path.dirname(os.path.abspath(__file__))
    
    normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomRotation(15), torchvision.transforms.ToTensor(), normalize])
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
        
    transform = transform_train
    if train is False:
        transform = transform_test

    mnistTrainSet = torchvision.datasets.CIFAR10(
        root=path + '/data', 
        train=train,
        download=True, 
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        mnistTrainSet,
        batch_size=bs, #256
        shuffle=True,
        num_workers=8)
    
    return loader