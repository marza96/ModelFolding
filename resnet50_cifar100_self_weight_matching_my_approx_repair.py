from model.resnet import ResNet50, merge_channel_ResNet50_clustering_approx_repair
from model.resnet import ResNet50, ResNet50Wider, merge_channel_ResNet50_clustering_approx_repair_wider, merge_channel_ResNet50_clustering_approx_repair

import wandb
import argparse
import torch
import os
import copy
from torchvision import datasets, transforms
from utils.utils import load_model

from tqdm import tqdm
import torchvision
import numpy as np

from math import pi
from thop import profile
from scipy.special import erf

import torch.nn.functional as F
import torch.nn as nn


class ConvBnormFuse(torch.nn.Module):
    def __init__(self, conv, bnorm):
        super().__init__()
        self.fused = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )
        self.weight = self.fused.weight
        self.bias = self.fused.bias

        self._fuse(conv, bnorm)

    def _fuse(self, conv, bn):
        w_conv = conv.weight.clone().reshape(conv.out_channels, -1).detach() #view umjesto reshape
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var))).detach()

        w_bn.requires_grad = False
        w_conv.requires_grad = False
        
        ww = torch.mm(w_bn.detach(), w_conv.detach())
        ww.requires_grad = False
        self.fused.weight.data = ww.data.view(self.fused.weight.detach().size()).detach() 
 
        if conv.bias is not None:
            b_conv = conv.bias.detach()
        else:
            b_conv = torch.zeros( conv.weight.size(0), device=conv.weight.device )

        bn.bias.requires_grad = False
        bn.weight.requires_grad = False

        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        
        bb = ( torch.matmul(w_bn, b_conv) + b_bn ).detach()
        self.fused.bias.data = bb.data

    def forward(self, x):
        return self.fused(x)


def fuse_bnorms(block, block_len):
    for i in range(block_len):
        alpha = block[i].bn1.weight.data.clone().detach()
        beta = block[i].bn1.bias.data.clone().detach()
        block[i].bn1.weight.data = torch.ones_like(block[i].bn1.weight.data)
        block[i].bn1.bias.data = torch.zeros_like(block[i].bn1.bias.data)

        block[i].conv1 = ConvBnormFuse(
            block[i].conv1,
            block[i].bn1
        ).fused
        block[i].bn1.weight.data = alpha
        block[i].bn1.bias.data = beta
        block[i].bn1.running_mean.data = torch.zeros_like(block[i].bn1.running_mean.data)
        block[i].bn1.running_var.data = torch.ones_like(block[i].bn1.running_var.data)


        alpha = block[i].bn2.weight.data.clone().detach()
        beta = block[i].bn2.bias.data.clone().detach()
        block[i].bn2.weight.data = torch.ones_like(block[i].bn2.weight.data)
        block[i].bn2.bias.data = torch.zeros_like(block[i].bn2.bias.data)

        block[i].conv2 = ConvBnormFuse(
            block[i].conv2,
            block[i].bn2
        ).fused
        block[i].bn2.weight.data = alpha
        block[i].bn2.bias.data = beta
        block[i].bn2.running_mean.data = torch.zeros_like(block[i].bn2.running_mean.data)
        block[i].bn2.running_var.data = torch.ones_like(block[i].bn2.running_var.data)


        alpha = block[i].bn3.weight.data.clone().detach()
        beta = block[i].bn3.bias.data.clone().detach()
        block[i].bn3.weight.data = torch.ones_like(block[i].bn3.weight.data)
        block[i].bn3.bias.data = torch.zeros_like(block[i].bn3.bias.data)

        block[i].conv3 = ConvBnormFuse(
            block[i].conv3,
            block[i].bn3
        ).fused
        block[i].bn3.weight.data = alpha
        block[i].bn3.bias.data = beta
        block[i].bn3.running_mean.data = torch.zeros_like(block[i].bn3.running_mean.data)
        block[i].bn3.running_var.data = torch.ones_like(block[i].bn3.running_var.data)

        if len(block[i].shortcut) == 2:
            alpha = block[i].shortcut[1].weight.data.clone().detach()
            beta = block[i].shortcut[1].bias.data.clone().detach()
            block[i].shortcut[1].weight.data = torch.ones_like(block[i].shortcut[1].weight.data)
            block[i].shortcut[1].bias.data = torch.zeros_like(block[i].shortcut[1].bias.data)
            block[i].shortcut[0] = ConvBnormFuse(
                block[i].shortcut[0],
                block[i].shortcut[1]
            ).fused
            block[i].shortcut[1].weight.data = alpha
            block[i].shortcut[1].bias.data = beta
            block[i].shortcut[1].running_mean.data = torch.zeros_like(block[i].shortcut[1].running_mean.data)
            block[i].shortcut[1].running_var.data = torch.ones_like(block[i].shortcut[1].running_var.data)


def fuse_bnorms_override(block, block_len):
    for i in range(block_len):
        alpha = block[i].bn1.weight.data.clone().detach()
        beta = block[i].bn1.bias.data.clone().detach()
        block[i].bn1.weight.data = torch.ones_like(block[i].bn1.weight.data)
        block[i].bn1.bias.data = torch.zeros_like(block[i].bn1.bias.data)

        block[i].conv1 = ConvBnormFuse(
            block[i].conv1,
            block[i].bn1
        ).fused
        block[i].bn1.weight.data = alpha
        block[i].bn1.bias.data = beta
        block[i].bn1.running_mean.data = torch.zeros_like(block[i].bn1.running_mean.data)
        block[i].bn1.running_var.data = torch.ones_like(block[i].bn1.running_var.data)


        alpha = block[i].bn2.weight.data.clone().detach()
        beta = block[i].bn2.bias.data.clone().detach()
        block[i].bn2.weight.data = torch.ones_like(block[i].bn2.weight.data)
        block[i].bn2.bias.data = torch.zeros_like(block[i].bn2.bias.data)

        block[i].conv2 = ConvBnormFuse(
            block[i].conv2,
            block[i].bn2
        ).fused
        block[i].bn2.weight.data = alpha
        block[i].bn2.bias.data = beta
        block[i].bn2.running_mean.data = torch.zeros_like(block[i].bn2.running_mean.data)
        block[i].bn2.running_var.data = torch.ones_like(block[i].bn2.running_var.data)


        alpha = block[i].bn3.weight.data.clone().detach()
        beta = block[i].bn3.bias.data.clone().detach()
        block[i].bn3.weight.data = torch.ones_like(block[i].bn3.weight.data)
        block[i].bn3.bias.data = torch.zeros_like(block[i].bn3.bias.data)

        block[i].conv3 = ConvBnormFuse(
            block[i].conv3,
            block[i].bn3
        ).fused
        block[i].bn3.weight.data = alpha
        block[i].bn3.bias.data = beta
        block[i].bn3.running_mean.data = torch.zeros_like(block[i].bn3.running_mean.data)
        block[i].bn3.running_var.data = torch.ones_like(block[i].bn3.running_var.data)

        if block[i].downsample is not None:
            alpha = block[i].downsample[1].weight.data.clone().detach()
            beta = block[i].downsample[1].bias.data.clone().detach()
            block[i].downsample[1].weight.data = torch.ones_like(block[i].downsample[1].weight.data)
            block[i].downsample[1].bias.data = torch.zeros_like(block[i].downsample[1].bias.data)
            block[i].downsample[0] = ConvBnormFuse(
                block[i].downsample[0],
                block[i].downsample[1]
            ).fused
            block[i].downsample[1].weight.data = alpha
            block[i].downsample[1].bias.data = beta
            block[i].downsample[1].running_mean.data = torch.zeros_like(block[i].downsample[1].running_mean.data)
            block[i].downsample[1].running_var.data = torch.ones_like(block[i].downsample[1].running_var.data)


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, figure_path, method, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    
    hooks = None
    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, threshold=threshold)
    
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

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



def get_datasets(train=True, bs=512): #8
    path   = os.path.dirname(os.path.abspath(__file__))
    
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    transform = transform_train
    if train is False:
        transform = transform_test

    mnistTrainSet = torchvision.datasets.CIFAR100(
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
    proj_name = "Resnet50 CIFAR100"
    
    # model = ResNet50(num_classes=100)
    # load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet50_1Xwider_CIFAR100.pt")
    # model.cuda()

    model = ResNet50Wider(num_classes=100, width_factor=2)
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet50_2Xwider_CIFAR100.pt", override=False)
    model.cuda()

    test_loader = get_datasets(train=False, bs=64)
    train_loader = get_datasets(train=True, bs=64)
    threshold=100.40
    figure_path = '/home/m/marza1/Iterative-Feature-Merging/'

    alpha = model.bn1.weight.data.clone().detach()
    beta = model.bn1.bias.data.clone().detach()
    model.bn1.weight.data = torch.ones_like(model.bn1.weight.data)
    model.bn1.bias.data = torch.zeros_like(model.bn1.bias.data)

    model.conv1 = ConvBnormFuse(
        model.conv1,
        model.bn1
    ).fused
    model.bn1.weight.data = alpha
    model.bn1.bias.data = beta
    model.bn1.running_mean.data = torch.zeros_like(model.bn1.running_mean.data)
    model.bn1.running_var.data = torch.ones_like(model.bn1.running_var.data)
    # fuse_bnorms(model.layer1, 3)
    # fuse_bnorms(model.layer2, 4)
    # fuse_bnorms(model.layer3, 6)
    # fuse_bnorms(model.layer4, 3)

    fuse_bnorms_override(model.layer1, 3)
    fuse_bnorms_override(model.layer2, 4)
    fuse_bnorms_override(model.layer3, 6)
    fuse_bnorms_override(model.layer4, 3)
    
    threshold=10.95
    figure_path = '/home/m/marza1/Iterative-Feature-Merging/'
    
    exp_name = "WM APPROX REPAIR 2x"
    desc = {"experiment": exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=exp_name
    )
    for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, threshold, figure_path, merge_channel_ResNet50_clustering_approx_repair_wider)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})

if __name__ == "__main__":
  main()

