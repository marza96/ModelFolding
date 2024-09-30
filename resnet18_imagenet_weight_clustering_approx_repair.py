
from model.resnet import merge_channel_ResNet18_big_clustering_approx_repair
from utils.utils import load_model, eval_model, fuse_bnorms_resnet18
from utils.datasets import get_imagenet
from torchvision.models import resnet18
from PIL import ImageFile
from thop import profile

import argparse
import torch
import wandb
import torch
import copy

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    
    if eval is True:
        return model, acc, param / origin_param

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--proj_name", type=str, help="", default="Folding ResNet18 ImageNet")
    parser.add_argument("--exp_name", type=str, help="", default="APPROX REPAIR")
    args = parser.parse_args()

    test_loader = get_imagenet(args.dataset_root, train=False, bs=64)
    train_loader = get_imagenet(args.dataset_root, train=True, bs=64)
    
    model = resnet18(num_classes=1000).to("cuda")
    load_model(model, args.checkpoint, override=False)

    fuse_bnorms_resnet18(model, override=False)

    desc = {"experiment": args.exp_name}
    wandb.init(
        project=args.proj_name,
        config=desc,
        name=args.exp_name
    )
    for ratio in [0.025, 0.05, 0.1, 0.12, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, merge_channel_ResNet18_big_clustering_approx_repair, eval=True)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": 1.0 - sparsity})


if __name__ == "__main__":
    main()