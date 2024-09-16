from model.resnet import ResNet18, merge_channel_ResNet18, merge_channel_ResNet18_clustering
from model.resnet import ResNet50, merge_channel_ResNet50_clustering
import wandb
import argparse
import torch
import os
import copy
from torchvision import datasets, transforms

from tqdm import tqdm
import torchvision
import numpy as np

from math import pi
from thop import profile
from scipy.special import erf

import torch.nn.functional as F
import torch.nn as nn



def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, figure_path, method, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    
    hooks = None
    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, threshold=threshold, hooks=hooks)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if eval is True:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = None

        model.train()
        for x, _ in tqdm(train_loader):
            model(x.to("cuda"))
            
        model.eval()

    # if eval is True:
    #     for module in model.modules():
    #         if isinstance(module, torch.nn.BatchNorm2d):
    #             module.reset_running_stats()
    #             module.momentum = None

    #     model.train()
    #     model(torch.load("input_tens.pt").to("cuda"))
    #     model.eval()

    if eval is True:
        correct = 0
        total_num = 0
        total_loss = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(tqdm(dataloader)):
                X = X.cuda()
                y = y.cuda()
                logit = model(X)
                loss = F.cross_entropy(logit, y)
                correct += (logit.max(1)[1] == y).sum().item()
                total_num += y.size(0)
                total_loss += loss.item()
        print(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}")

    print(
        f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    
    if eval is True:
        return model, correct/total_num, param / origin_param

    return model


def load_model(model, i):
    sd = torch.load(i, map_location=torch.device('cpu'))
    new_sd = copy.deepcopy(sd)

    for key, value in sd.items():
        if "downsample" in key:
            new_key = key.replace("downsample", "shortcut")
            new_sd[new_key] = value
            new_sd.pop(key)

        if "fc" in key:
            new_key = key.replace("fc", "linear")
            new_sd[new_key] = value
            new_sd.pop(key)

    model.load_state_dict(new_sd)


def get_datasets(train=True, bs=256): #8
    path   = os.path.dirname(os.path.abspath(__file__))
    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        
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


def main():
    proj_name = "IFM-vs-Weight-Clustering-Resnet18-REPAIR"
    
    model = ResNet50()
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet50_1Xwider_CIFAR10.pt")
    model.cuda()

    test_loader = get_datasets(train=False)
    train_loader = get_datasets(train=True)
    max_ratio=0.42#0.35
    threshold=100.40
    figure_path = '/home/m/marza1/Iterative-Feature-Merging/'

    # correct = 0
    # total_num = 0
    # total_loss = 0
    # with torch.no_grad():
    #     for i, (X, y) in enumerate(tqdm(test_loader)):
    #         X = X.cuda()
    #         y = y.cuda()
    #         logit = model(X)
    #         loss = F.cross_entropy(logit, y)
    #         correct += (logit.max(1)[1] == y).sum().item()
    #         total_num += y.size(0)
    #         total_loss += loss.item()
    # print(f"model before adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}")
    
    
    total_params = sum(p.numel() for p in model.parameters())
    new_model, _, _ = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, 0.5, threshold, figure_path, merge_channel_ResNet50_clustering, eval=True)
    new_total_params = sum(p.numel() for p in new_model.parameters())
    print("ACT SP", new_total_params / total_params)

    # exp_name = "Weight-Clustering Approx REPAIR"
    # desc = {"experiment": exp_name}
    # wandb.init(
    #     project=proj_name,
    #     config=desc,
    #     name=exp_name
    # )
    # for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
    #     new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, threshold, figure_path, merge_channel_ResNet18_clustering, eval=True)
    #     wandb.log({"test acc": acc})
    #     wandb.log({"sparsity": sparsity})

    # exp_name = "IFM"
    # desc = {"experiment": exp_name}
    # wandb.init(
    #     project=proj_name,
    #     config=desc,
    #     name=exp_name,
    #     reinit=True
    # )
    # # for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
    # for ratio in [0.05, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    #     new_model, acc, sparsity  = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, threshold, figure_path, merge_channel_ResNet18)
    #     wandb.log({"test acc": acc})
    #     wandb.log({"sparsity": sparsity})

if __name__ == "__main__":
  main()
