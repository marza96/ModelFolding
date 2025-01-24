from model.vgg import merge_channel_vgg11_clustering, merge_channel_vgg11_clustering_approx_repair
from torchvision.models.vgg import make_layers, VGG
from utils.datasets import get_cifar10
from utils.utils import fuse_bnorms_vgg, eval_model
from utils.utils import DF_REPAIR, DI_REPAIR, NO_REPAIR, REPAIR

import argparse
import torch
import wandb
import copy
from tqdm import tqdm
from thop import profile


def test_merge(origin_model, checkpoint, test_loader, train_loader, method, repair, ratio, di_samples_path):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    _, origin_param = profile(origin_model, inputs=(input,))
    model, _ = method(origin_model, checkpoint, max_ratio=ratio)
    model.cuda()
    model.eval()
    _, param = profile(model, inputs=(input,))
    
    if repair != NO_REPAIR and repair != DF_REPAIR:
        if repair == REPAIR:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            for x, _ in tqdm(train_loader):
                model(x.to("cuda"))

        elif repair == DI_REPAIR: 
            model.eval()

            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            model(torch.load(di_samples_path).to("cuda"))

    model.eval()
    acc, loss = eval_model(model, test_loader)

    print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")

    return acc, param / origin_param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--repair", type=str, default="NO_REPAIR", help="")
    parser.add_argument("--proj_name", type=str, help="", default="Folding VGG11 cifar10")
    parser.add_argument("--exp_name", type=str, help="", default="WM REPAIR")
    parser.add_argument("--di_samples_path", type=str, default="cifar10_vgg.pt")
    args = parser.parse_args()

    vgg11_cfg  = [args.width * 64, 'M', args.width * 128, 'M', args.width * 256, args.width * 256, 'M', args.width * 512, args.width * 512, 'M', args.width * 512, 512, 'M']
    features   = make_layers(vgg11_cfg, batch_norm=True)
    model      = VGG(features=features, num_classes=10)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model.load_state_dict(checkpoint)
    model.cuda()
    
    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)

    method = merge_channel_vgg11_clustering
    if args.repair == DF_REPAIR:
        fuse_bnorms_vgg(model)
        method = merge_channel_vgg11_clustering_approx_repair

    model.eval()

    exp_name = args.exp_name
    desc = {"experiment": exp_name}
    wandb.init(
        project=args.proj_name,
        config=desc,
        name=exp_name,
        reinit=True
    )
    
    for ratio in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader,  method, args.repair, ratio, args.di_samples_path)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()
