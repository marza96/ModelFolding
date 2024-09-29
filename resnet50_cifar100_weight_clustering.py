from model.resnet import ResNet50, ResNet50Wider, merge_channel_ResNet50_clustering_wider, merge_channel_ResNet50_clustering

import wandb
import argparse
import torch
import os
import copy
from torchvision import datasets, transforms
from utils.utils import load_model

from tqdm import tqdm
import torchvision

from thop import profile

import torch.nn.functional as F

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
    #     model(torch.load("cifar100.pt").to("cuda"))
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
    model = ResNet50Wider(num_classes=100, width_factor=2)
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet50_2Xwider_CIFAR100.pt", override=False)
    model.cuda()

    test_loader = get_datasets(train=False, bs=64)
    train_loader = get_datasets(train=True, bs=64)


    threshold=10.95
    figure_path = '/home/m/marza1/Iterative-Feature-Merging/'
    
    exp_name = "WM REPAIR 2x"
    desc = {"experiment": exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=exp_name
    )
    for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, threshold, figure_path, merge_channel_ResNet50_clustering_wider)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()
