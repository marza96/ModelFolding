from model.vgg import merge_channel_vgg11_clustering_approx_repair
from torchvision.models.vgg import VGG, make_layers
from utils.utils import load_model, eval_model, fuse_bnorms_vgg
from utils.datasets import get_imagenet
from PIL import ImageFile
from thop import profile

import argparse
import torch
import copy
import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test_merge(origin_model, checkpoint, test_loader, train_loader, ratio, method):
    input = torch.torch.randn(1, 3, 224, 224).cuda()
    origin_model.cuda()
    origin_model.eval()

    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    model, _ = method(origin_model, checkpoint, max_ratio=ratio)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))
    
    acc, loss = eval_model(model, test_loader)

    print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")
    
    return acc, param / origin_param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--proj_name", type=str, help="", default="Folding vgg11 ImageNet")
    parser.add_argument("--exp_name", type=str, help="", default="APPROX REPAIR")
    args = parser.parse_args()

    vgg11_cfg  = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    features   = make_layers(vgg11_cfg, batch_norm=True)
    model      = VGG(features=features, num_classes=1000)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model.load_state_dict(checkpoint)
    model.cuda()

    fuse_bnorms_vgg(model)
    test_loader = get_imagenet(args.dataset_root, train=False, bs=64)
    train_loader = get_imagenet(args.dataset_root, train=True, bs=64)

    proj_name = args.proj_name

    desc = {"experiment": args.exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=args.exp_name,
        reinit=True
    )

    for ratio in [0.025, 0.05, 0.1, 0.12, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, merge_channel_vgg11_clustering_approx_repair)
        print("ACC", acc, sparsity)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()
