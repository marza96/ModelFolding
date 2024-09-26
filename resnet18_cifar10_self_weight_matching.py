from model.resnet import ResNet18, merge_channel_ResNet18_clustering
from utils.datasets import get_cifar10

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

from thop import profile

import torch.nn.functional as F


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, figure_path, method, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    var = None
    origin_var = None

    # origin_var = measure_avg_var(origin_model, dataloader)
    # print(origin_var)

    origin_flop, origin_param = profile(origin_model, inputs=(input,))

    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, threshold=threshold, hooks=None)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    # if eval is True:
    #     for module in model.modules():
    #         if isinstance(module, torch.nn.BatchNorm2d):
    #             module.reset_running_stats()
    #             module.momentum = None

    #     model.train()
    #     model(torch.load("cifar10.pt").to("cuda"))
    #     model.eval()


    if eval is True:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = None

        model.train()
        for x, _ in tqdm(train_loader):
            model(x.to("cuda"))
            
        model.eval()
    
    # var = measure_avg_var(model, dataloader)
    
    # ratios = list()
    # for i in range(len(var)):
    #     ratios.append(float((var[i] / origin_var[i])))

    # print(ratios)

    # if eval is True:
    #     for module in model.modules():
    #         if isinstance(module, torch.nn.BatchNorm2d):
    #             module.reset_running_stats()
    #             module.momentum = None

    #     model.train()
    #     for x, _ in tqdm(train_loader):
    #         model(x.to("cuda"))
            
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
    # handles.append(model.conv1.register_forward_hook(hooks["bn1"]))
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

def main():
    proj_name = "resnet18 CIFAR10 WM 2 blocks"
    
    model = ResNet18()
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet18_1Xwider_CIFAR10_latest.pt")
    model.cuda()

    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)
    max_ratio=0.42#0.35
    threshold=100.40
    figure_path = '/home/m/marza1/Iterative-Feature-Merging/'

    exp_name = "Weight-Clustering REPAIR"
    desc = {"experiment": exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=exp_name
    )
    for ratio in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity, var_ratio = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, threshold, figure_path, merge_channel_ResNet18_clustering, eval=True)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": 1.0 - sparsity})
        wandb.log({"var_ratio": var_ratio})


if __name__ == "__main__":
  main()
