from model.vgg import merge_channel_vgg11_clustering_approx_repair
from torchvision.models.vgg import VGG, make_layers
from utils.utils import eval_model, fuse_bnorms_vgg
from utils.datasets import get_cifar10
from thop import profile

import argparse
import torch
import copy
import wandb


def test_merge(origin_model, checkpoint, test_loader, train_loader, max_ratio, threshold, method):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    _, origin_param = profile(origin_model, inputs=(input,))
    model, _ = method(origin_model, checkpoint, max_ratio=max_ratio, threshold=threshold)
    model.cuda()
    model.eval()
    _, param = profile(model, inputs=(input,))

    acc, loss = eval_model(model, test_loader)

    print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")

    return acc, param / origin_param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--proj_name", type=str, help="", default="Folding VGG11 cifar10")
    parser.add_argument("--exp_name", type=str, help="", default="WM REPAIR")
    args = parser.parse_args()

    vgg11_cfg  = [args.width * 64, 'M', args.width * 128, 'M', args.width * 256, args.width * 256, 'M', args.width * 512, args.width * 512, 'M', args.width * 512, 512, 'M']
    features   = make_layers(vgg11_cfg, batch_norm=True)
    model      = VGG(features=features, num_classes=10)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model.load_state_dict(checkpoint)
    model.cuda()

    fuse_bnorms_vgg(model)
    test_loader  = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)

    proj_name = args.proj_name
    desc      = {"experiment": args.exp_name}

    wandb.init(
        project=proj_name,
        config=desc,
        name=args.exp_name,
        reinit=True
    )
    for ratio in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, 100.0,  merge_channel_vgg11_clustering_approx_repair)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()
