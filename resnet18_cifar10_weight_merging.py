from model.resnet import ResNet18, fuse_channel_resnet18_clustering
from utils.utils import load_model, eval_model, LayerActivationHook
from utils.datasets import get_cifar10

import numpy as np
import argparse
import torch
import copy
from tqdm import tqdm
from thop import profile


def test_merge(origin_model_a, checkpoint_a, origin_model_b, checkpoint_b, test_loader, train_loader, method):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model_a.cuda()
    origin_model_a.eval()
    origin_model_b.cuda()
    origin_model_b.eval()

    _, origin_param = profile(origin_model_a, inputs=(input,))
    model, _ = method(
        origin_model_a, 
        origin_model_b, 
        checkpoint_a, 
        checkpoint_b, 
        regularizer=0.01
    )
    
    model.cuda()
    model.eval()
    _, param = profile(model, inputs=(input,))

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None

    model.train()
    for x, _ in tqdm(train_loader, desc="REPAIR"):
        model(x.to("cuda"))

    model.eval()
    acc, _ = eval_model(model, test_loader)

    print(f"model after adapt: acc:{acc * 100:.2f}")

    return acc, param / origin_param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", type=str)
    parser.add_argument("--checkpoint_b", type=str)

    args = parser.parse_args()

    model_a = ResNet18(num_classes=10)
    model_b = ResNet18(num_classes=10)

    load_model(
        model_a, 
        args.checkpoint_a, 
        mapping={
            "downsample": "shortcut",
            "fc": "linear"
        }
    )

    load_model(
        model_b, 
        args.checkpoint_b, 
        mapping={
            "downsample": "shortcut",
            "fc": "linear"
        }
    )

    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True, bs=1024)

    model_a.eval()
    model_b.eval()

    test_merge(copy.deepcopy(model_a), copy.deepcopy(model_a).state_dict(), copy.deepcopy(model_b), copy.deepcopy(model_b).state_dict(), test_loader, train_loader,  fuse_channel_resnet18_clustering)


if __name__ == "__main__":
  main()
