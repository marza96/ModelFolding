from model.resnet import ResNet50, merge_channel_ResNet50_clustering
from utils.utils import DI_REPAIR, NO_REPAIR, REPAIR
from utils.utils import eval_model, load_model
from utils.datasets import get_cifar10
from thop import profile
from tqdm import tqdm

import wandb
import argparse
import torch
import copy


def test_merge(origin_model, checkpoint, test_loader, train_loader, ratio, method, repair, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    
    hooks = None
    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=ratio)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if eval is True:
        if repair != NO_REPAIR:
            if repair == DI_REPAIR: 
                for module in model.modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.reset_running_stats()
                        module.momentum = None

                model.train()
                model(torch.load("cifar10.pt").to("cuda"))
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

        acc, loss = eval_model(model, test_loader)
        print(f"model after adapt: acc:{acc:.2f}%, avg loss:{loss:.4f}")
        print(f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    
        return model, acc, param / origin_param

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--repair", type=str, default="NO_REPAIR", help="")
    parser.add_argument("--proj_name", type=str, help="", default="Folding Resnet50 cifar10")
    parser.add_argument("--exp_name", type=str, help="", default="WM REPAIR")
    args = parser.parse_args()

    model = ResNet50()
    load_model(model, args.checkpoint)
    model.cuda()

    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)

    desc = {"experiment": args.exp_name}
    wandb.init(
        project=args.proj_name,
        config=desc,
        name=args.exp_name
    )
    for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, merge_channel_ResNet50_clustering, args.repair, eval=True)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()
