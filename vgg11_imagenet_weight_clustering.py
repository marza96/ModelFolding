from model.vgg import merge_channel_vgg11_clustering, merge_channel_vgg11_clustering_approx_repair
from utils.utils import eval_model, fuse_bnorms_vgg
from utils.utils import DI_REPAIR, REPAIR, NO_REPAIR, DF_REPAIR
from utils.datasets import get_imagenet

import argparse
import torch
import wandb
import copy
from tqdm import tqdm
from thop import profile
from torchvision.models.vgg import VGG, make_layers
from utils.datasets import get_imagenet
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test_merge(origin_model, checkpoint, test_loader, train_loader, max_ratio, method, repair, di_samples_path):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    model, _ = method(origin_model, checkpoint, max_ratio=max_ratio)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))
    
    if repair != NO_REPAIR and repair != DF_REPAIR:
        if repair == DI_REPAIR:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            data = torch.load(di_samples_path, map_location="cpu")
            model.train()
            model(data[:128, :, :, :].cuda())
            model(data[128:, :, :, :].cuda())
            model.eval()

        elif repair == REPAIR:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None
            
            model.train()
            idx = 0
            for x, _ in tqdm(train_loader):
                model(x.to("cuda"))
                idx += 1
                if idx == 4:
                    break
                
            model.eval()

    acc, loss = eval_model(model, test_loader)
    print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")

    return acc, param / origin_param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--repair", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--proj_name", type=str, help="", default="Folding ResNet18 ImageNet")
    parser.add_argument("--exp_name", type=str, help="", default="APPROX REPAIR")
    parser.add_argument("--di_samples_path", type=str, default="di_vgg_imagenet_256.pt")
    args = parser.parse_args()

    # load model
    vgg11_cfg  = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    features   = make_layers(vgg11_cfg, batch_norm=True)
    model      = VGG(features=features, num_classes=1000)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.cuda()

    test_loader = get_imagenet(args.dataset_root, train=False, bs=64)
    train_loader = get_imagenet(args.dataset_root, train=True, bs=64)

    method = merge_channel_vgg11_clustering
    if args.repair == DF_REPAIR:
        fuse_bnorms_vgg(model)
        method = merge_channel_vgg11_clustering_approx_repair

    desc = {"experiment": args.exp_name}
    wandb.init(
        project=args.proj_name,
        config=desc,
        name=args.exp_name,
        reinit=True
    )

    for ratio in [0.025, 0.05, 0.1, 0.12, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, method, args.repair, args.di_samples_path)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()
