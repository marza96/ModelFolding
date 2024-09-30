
from model.resnet import merge_channel_ResNet18_big_clustering
from utils.datasets import get_imagenet
from utils.utils import load_model, eval_model
from utils.utils import DF_REPAIR, DI_REPAIR, NO_REPAIR, REPAIR

from torchvision.models import resnet18
from tqdm import tqdm
from PIL import ImageFile
from thop import profile

import copy
import torch
import wandb
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, method, repair, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if eval is True:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = None

        model.train()
        print("DI")
        model(torch.load("imagenet_di_512.pt").to("cuda"))
        model.eval()

    if eval is True:
        acc, loss = eval_model(model, dataloader)

        print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")
        print(f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    
        return model, acc, param / origin_param

    return model


def main():
    test_loader = get_imagenet("/home/m/marza1/imagenet/ImageNet/ILSVRC12", train=False, bs=512)
    train_loader = get_imagenet("/home/m/marza1/imagenet/ImageNet/ILSVRC12", train=True, bs=256)

    model = resnet18(num_classes=1000).to("cuda")
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet18_imagenet.pt", override=False)

    proj_name = "CLUSTERING Imagenet New"

    exp_name = "Weight-Clustering NO REPAIR"
    desc = {"experiment": exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=exp_name
    )
    for ratio in [0.025, 0.05, 0.1, 0.12, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, 100.0, "figure_path", merge_channel_ResNet18_big_clustering, eval=True)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": 1.0 - sparsity})


if __name__ == "__main__":
    main()