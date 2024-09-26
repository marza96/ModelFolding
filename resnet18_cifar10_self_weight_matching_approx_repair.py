from model.resnet import ResNet18, merge_channel_ResNet18_clustering_approx_repair

import wandb
import argparse
import torch
import os
import copy
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

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


def fuse_bnorms(block):
    for i in range(2):
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

class AvgLayerStatisticsHook:
    def __init__(self, conv=False):
        self.conv = conv
        self.bnorm = None

    def __call__(self, module, input, output):
        if self.bnorm is None:
            if self.conv is True:
                self.bnorm = torch.nn.BatchNorm2d(output.shape[1]).to("cuda")
            else:
                self.bnorm = torch.nn.BatchNorm1d(output.shape[1]).to("cuda")

            self.bnorm.train()
            self.bnorm.momentum = None
        
        self.bnorm(output)

    def get_stats(self):
        return self.bnorm.running_var.mean()


def measure_avg_var(model, dataloader):
    hooks = {
        # "bn1": AvgLayerStatisticsHook(conv=True),
        # "relu_conv": AvgLayerStatisticsHook(conv=True),
        # "layer1.0.relu1": AvgLayerStatisticsHook(conv=True),
        # "layer1.1.relu1": AvgLayerStatisticsHook(conv=True),
        # "layer2.0.relu1": AvgLayerStatisticsHook(conv=True),
        # "layer2.0.relu2": AvgLayerStatisticsHook(conv=True),
        # "layer2.1.relu1": AvgLayerStatisticsHook(conv=True),
        # "layer3.0.relu1": AvgLayerStatisticsHook(conv=True),
        # "layer3.0.relu2": AvgLayerStatisticsHook(conv=True),
        # "layer3.1.relu1": AvgLayerStatisticsHook(conv=True),
        # "layer4.0.relu1": AvgLayerStatisticsHook(conv=True),
        # "layer4.0.relu2": AvgLayerStatisticsHook(conv=True),
        "layer4.1.relu1": AvgLayerStatisticsHook(conv=True)
    }

    handles = list()
    # handles.append(model.bn1.register_forward_hook(hooks["bn1"]))
    # handles.append(model.conv1.register_forward_hook(hooks["relu_conv"]))
    # handles.append(model.layer1[0].conv1.register_forward_hook(hooks["layer1.0.relu1"]))
    # handles.append(model.layer1[1].conv1.register_forward_hook(hooks["layer1.1.relu1"]))
    # handles.append(model.layer2[0].conv1.register_forward_hook(hooks["layer2.0.relu1"]))
    # handles.append(model.layer2[0].conv2.register_forward_hook(hooks["layer2.0.relu2"]))
    # handles.append(model.layer2[1].conv1.register_forward_hook(hooks["layer2.1.relu1"]))
    # handles.append(model.layer3[0].conv1.register_forward_hook(hooks["layer3.0.relu1"]))
    # handles.append(model.layer3[0].conv2.register_forward_hook(hooks["layer3.0.relu2"]))
    # handles.append(model.layer3[1].conv1.register_forward_hook(hooks["layer3.1.relu1"]))
    # handles.append(model.layer4[0].conv1.register_forward_hook(hooks["layer4.0.relu1"]))
    # handles.append(model.layer4[0].conv2.register_forward_hook(hooks["layer4.0.relu2"]))
    handles.append(model.layer4[1].conv1.register_forward_hook(hooks["layer4.1.relu1"]))

    model.cuda()
    model.eval()

    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(tqdm(dataloader)), desc="Eval initial stats"):
            X = X.cuda()
            y = y.cuda()
            _ = model(X)

    avg_vars = list()
    for key in hooks.keys():
        print("KKK", key)
        avg_vars.append(hooks[key].get_stats())

    for handle in handles:
        handle.remove()

    return avg_vars


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, figure_path, method, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_var = None
    var = None

    # origin_var = measure_avg_var(origin_model, dataloader)
    # print(origin_var)

    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    
    hooks = None

    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, threshold=threshold, hooks=hooks)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))
    
    # var = measure_avg_var(model, dataloader)
    # ratios = list()
    # for i in range(len(var)):
    #     ratios.append(float((var[i] / origin_var[i])))
    # print(ratios)

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
        if var is not None:
            return model, correct/total_num, param / origin_param, float((var[-1] / origin_var[-1]))
        
        return model, correct/total_num, param / origin_param, -1

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


def get_datasets(train=True, bs=512): #8
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
        batch_size=bs, 
        shuffle=True,
        num_workers=8)
    
    return loader


def main():
    # proj_name = "WM-approx-repair"
    proj_name = "resnet18 CIFAR10 WM 2 blocks"
    
    model = ResNet18()
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet18_1Xwider_CIFAR10_latest.pt")
    model.cuda()

    test_loader = get_datasets(train=False)
    train_loader = get_datasets(train=True)

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
    fuse_bnorms(model.layer1)
    fuse_bnorms(model.layer2)
    fuse_bnorms(model.layer3)
    fuse_bnorms(model.layer4)

    threshold=100.40
    figure_path = '/home/m/marza1/Iterative-Feature-Merging/'

    # total_params = sum(p.numel() for p in model.parameters())
    # new_model, _, _ = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, 0.55, threshold, figure_path, merge_channel_ResNet18_clustering_approx_repair, eval=True)
    # new_total_params = sum(p.numel() for p in new_model.parameters())
    # print("ACT SP", new_total_params / total_params)

    # return 

    exp_name = "Weight-Clustering APPROXIMATE REPAIR"
    desc = {"experiment": exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=exp_name
    )
    for ratio in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity, var_ratio = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, threshold, figure_path, merge_channel_ResNet18_clustering_approx_repair, eval=True)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": 1.0 - sparsity})
        wandb.log({"var_ratio": var_ratio})

if __name__ == "__main__":
  main()
