from model.vgg import merge_channel_vgg11_clustering_approx_repair, get_axis_to_perm
import argparse
import torch
import os
from torchvision import datasets, transforms
from tqdm import tqdm
import copy
from thop import profile
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models.vgg import VGG, make_layers
import wandb
from utils.datasets import get_cifar10
from utils.utils import ConvBnormFuse


def fuse_bnorms(model):
    for i in range(len(model.features)):
        if isinstance(model.features[i], torch.nn.Conv2d):
            print(type(model.features[i]), model.features[i + 1])

            alpha = model.features[i + 1].weight.data.clone().detach()
            beta = model.features[i + 1].bias.data.clone().detach()
            model.features[i + 1].weight.data = torch.ones_like(model.features[i + 1].weight.data)
            model.features[i + 1].bias.data = torch.zeros_like(model.features[i + 1].bias.data)

            model.features[i] = ConvBnormFuse(
                model.features[i],
                model.features[i + 1]
            ).fused
            model.features[i + 1].weight.data = alpha
            model.features[i + 1].bias.data = beta
            model.features[i + 1].running_mean.data = torch.zeros_like(model.features[i + 1].running_mean.data)
            model.features[i + 1].running_var.data = torch.ones_like(model.features[i + 1].running_var.data)


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, method):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    model, _ = method(origin_model, checkpoint, max_ratio=max_ratio, threshold=threshold)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))
    
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

    return correct / total_num, param / origin_param



def main():
    vgg11_cfg       = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    features = make_layers(vgg11_cfg, batch_norm=True)
    model = VGG(features=features, num_classes=10)
    checkpoint = torch.load("/home/m/marza1/Iterative-Feature-Merging/vgg11_bn_1Xwider_CIFAR10.pt")

    model.load_state_dict(checkpoint)
    model.cuda()

    fuse_bnorms(model)
    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)

    proj_name = "Folding VGG11 cifar10"

    exp_name = "WM APPROX REPAIR"
    desc = {"experiment": exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=exp_name,
        reinit=True
    )
    for ratio in [0.15, 0.16, 0.20, 0.25, 0.27, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95]:
        acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, 100.0,  merge_channel_vgg11_clustering_approx_repair)
        print("ACC", acc)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})

if __name__ == "__main__":
  main()
