from model.resnet import ResNet18, merge_channel_ResNet18_clustering, merge_channel_ResNet18_clustering_approx_repair
from utils.utils import eval_model, load_model, fuse_bnorms_arbitrary_resnet, AvgLayerStatisticsHook
from utils.utils import DI_REPAIR, NO_REPAIR, REPAIR, DF_REPAIR
from utils.datasets import get_cifar10
from thop import profile
from tqdm import tqdm

import wandb
import argparse
import torch
import copy


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, method, repair, di_samples_path, eval=True, measure_variance=False):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    origin_flop, origin_param = profile(origin_model, inputs=(input,))

    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, hooks=None)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if repair != NO_REPAIR and repair != DF_REPAIR:
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

        return model, acc, param / origin_param, -1

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--repair", type=str, default="NO_REPAIR", help="")
    parser.add_argument("--proj_name", type=str, help="", default="Folding Resnet18 cifar10")
    parser.add_argument("--exp_name", type=str, help="", default="WM REPAIR")
    parser.add_argument("--di_samples_path", type=str, default="cifar10.pt")
    args = parser.parse_args()

    model = ResNet18()
    load_model(model, args.checkpoint)
    model.cuda()

    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)

    method = merge_channel_ResNet18_clustering
    if args.repair == DF_REPAIR:
        fuse_bnorms_arbitrary_resnet(model, [2,2,2,2], override=False)
        method = merge_channel_ResNet18_clustering_approx_repair

    desc = {"experiment": args.exp_name}
    wandb.init(
        project=args.proj_name,
        config=desc,
        name=args.exp_name
    )
    for ratio in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity, var_ratio = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, method, args.repair, args.di_samples_path)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": 1.0 - sparsity})
        wandb.log({"var_ratio": var_ratio})


if __name__ == "__main__":
  main()
