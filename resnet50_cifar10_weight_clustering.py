from model.resnet import ResNet50, ResNet50Wider, merge_channel_ResNet50_clustering_wider, merge_channel_ResNet50_clustering, merge_channel_ResNet50_clustering_approx_repair_wider
from utils.utils import load_model, eval_model, fuse_bnorms_arbitrary_resnet
from utils.utils import REPAIR, DI_REPAIR, NO_REPAIR, DF_REPAIR
from utils.datasets import get_cifar10
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
        acc, loss = eval_model(model, test_loader)
        print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")

        print(
            f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    
        return model, acc, param / origin_param

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--repair", type=str, default="NO_REPAIR", help="")
    parser.add_argument("--proj_name", type=str, help="", default="Folding Resnet50 cifar10")
    parser.add_argument("--exp_name", type=str, help="", default="WM REPAIR")
    parser.add_argument("--di_samples_path", type=str, default="cifar10.pt")
    args = parser.parse_args()
    
    model = ResNet50Wider(num_classes=10, width_factor=args.width)
    load_model(model, args.checkpoint, override=False)
    model.cuda()

    test_loader = get_cifar10(train=False, bs=128)
    train_loader = get_cifar10(train=True, bs=256)

    method = merge_channel_ResNet50_clustering_wider
    if args.repair == DF_REPAIR:
        fuse_bnorms_arbitrary_resnet(model, [3, 4, 6, 3], override=True)
        method = merge_channel_ResNet50_clustering_approx_repair_wider

    proj_name = args.proj_name
    desc = {"experiment": args.exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=args.exp_name
    )

    accs = list()
    sps = list()
    for ratio in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, method, args.repair, args.di_samples_path)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})
        accs.append(acc)
        sps.append(sparsity)


if __name__ == "__main__":
  main()
