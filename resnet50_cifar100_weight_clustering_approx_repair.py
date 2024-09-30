from model.resnet import ResNet50, ResNet50Wider, merge_channel_ResNet50_clustering_approx_repair_wider, merge_channel_ResNet50_clustering_approx_repair
from utils.utils import load_model, eval_model, fuse_bnorms_arbitrary_resnet
from utils.datasets import get_cifar100
from thop import profile

import wandb
import argparse
import torch
import copy


def test_merge(origin_model, checkpoint, test_loader, train_loader, ratio, method, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    
    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=ratio)
    
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if eval is True:
        acc, loss = eval_model(model, test_loader)
        print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")
        print(f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    
        return model, acc, param / origin_param

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--proj_name", type=str, help="", default="Folding Resnet50 cifar100")
    parser.add_argument("--exp_name", type=str, help="", default="WM APPROX REPAIR")
    args = parser.parse_args()

    model = ResNet50Wider(num_classes=100, width_factor=args.width)
    #"/home/m/marza1/Iterative-Feature-Merging/resnet50_2Xwider_CIFAR100.pt"
    load_model(model, args.checkpoint, override=False)
    model.cuda()

    test_loader = get_cifar100(train=False, bs=64)
    train_loader = get_cifar100(train=True, bs=64)

    fuse_bnorms_arbitrary_resnet(model, [3, 4, 6, 3], override=True)
    
    desc = {"experiment": args.exp_name}
    wandb.init(
        project=args.proj_name,
        config=desc,
        name=args.exp_name
    )
    for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, merge_channel_ResNet50_clustering_approx_repair_wider)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()

