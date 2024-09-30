from model.resnet import ResNet18, merge_channel_ResNet18_clustering
from utils.utils import load_model, eval_model
from utils.datasets import get_cifar100
from utils.utils import DF_REPAIR, DI_REPAIR, NO_REPAIR, REPAIR

import wandb
import argparse
import torch
import copy

from tqdm import tqdm
from thop import profile

import torch.nn.functional as F


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, method, repair, di_samples_path, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    
    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if repair != NO_REPAIR:
        if repair == DI_REPAIR: 
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            model(torch.load(di_samples_path).to("cuda"))
            model.eval()

        elif repair == REPAIR:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            for x, _ in tqdm(train_loader):
                model(x.to("cuda"))
                
            model.eval()

    if eval is True:
        acc, loss = eval_model(model, dataloader)
        print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")

        print(
            f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
                
        return model, acc, param / origin_param

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--repair", type=str, default="REPAIR", help="")
    parser.add_argument("--proj_name", type=str, help="", default="Folding Resnet18 cifar100")
    parser.add_argument("--exp_name", type=str, help="", default="WM REPAIR")
    parser.add_argument("--di_samples_path", type=str, default="cifar100.pt")
    args = parser.parse_args()

    proj_name = args.proj_name
    
    model = ResNet18(num_classes=100)
    #"/home/m/marza1/Iterative-Feature-Merging/resnet18_1Xwider_CIFAR100.pt"
    load_model(model, args.checkpoint)
    model.cuda()

    test_loader = get_cifar100(train=False)
    train_loader = get_cifar100(train=True)

    desc = {"experiment": args.exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=args.exp_name
    )
    for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, merge_channel_ResNet18_clustering, args.repair, args.di_samples_path)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()
