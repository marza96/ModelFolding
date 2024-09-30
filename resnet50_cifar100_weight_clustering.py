from model.resnet import ResNet50, ResNet50Wider, merge_channel_ResNet50_clustering_wider, merge_channel_ResNet50_clustering
from utils.utils import load_model, eval_model
from utils.utils import REPAIR, DI_REPAIR, NO_REPAIR
from utils.datasets import get_cifar100
from thop import profile
from tqdm import tqdm

import wandb
import argparse
import torch
import copy


def test_merge(origin_model, checkpoint, test_loader, train_loader, max_ratio, method, repair, di_samples_path, eval=True):
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
                break
                
            model.eval()

        acc, loss = eval_model(model, test_loader)

        return model, acc, param / origin_param

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--repair", type=str, default="NO_REPAIR", help="")
    parser.add_argument("--proj_name", type=str, help="", default="Folding Resnet50 cifar100")
    parser.add_argument("--exp_name", type=str, help="", default="WM REPAIR")
    parser.add_argument("--di_samples_path", type=str, default="cifar100.pt")
    args = parser.parse_args()
    
    model = ResNet50Wider(num_classes=100, width_factor=args.width)
    load_model(model, args.checkpoint, override=False)
    model.cuda()

    test_loader = get_cifar100(train=False, bs=64)
    train_loader = get_cifar100(train=True, bs=64)

    proj_name = args.proj_name
    
    desc = {"experiment": args.exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=args.exp_name
    )
    for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, merge_channel_ResNet50_clustering_wider, args.repair, args.di_samples_path)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()
