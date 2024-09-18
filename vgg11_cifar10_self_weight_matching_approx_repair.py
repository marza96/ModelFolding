from model.vgg import merge_channel_vgg16, get_axis_to_perm
import argparse
import torch
import os
from torchvision import datasets, transforms
from tqdm import tqdm
import copy
from thop import profile
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models.vgg import VGG, make_layers
from utils.utils import ConvBnormFuse
import torchvision


def fuse_bnorms(model):
    for i in range(len(model.features)):
        if isinstance(model.features[i], torch.nn.Conv2d):
            print(type(model.features[i]), model.features[i + 1])

            alpha = model.features[i + 1].weight.data.clone().detach()
            beta = model.features[i + 1].bias.data.clone().detach()
            model.features[i + 1].weight.data = torch.ones_like(model.features[i + 1].weight.data)
            model.features[i + 1].bias.data = torch.zeros_like(model.features[i + 1].bias.data)

            model.features[i] = ConvBnormFuse(
                model.features[i],
                model.features[i + 1]
            ).fused
            model.features[i + 1].weight.data = alpha
            model.features[i + 1].bias.data = beta
            model.features[i + 1].running_mean.data = torch.zeros_like(model.features[i + 1].running_mean.data)
            model.features[i + 1].running_var.data = torch.ones_like(model.features[i + 1].running_var.data)


def test_activation_cluster(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, figure_path, vgg_name):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    get_axis_to_perm(origin_model)
    model, _ = merge_channel_vgg16(origin_model, vgg_name, checkpoint, max_ratio=max_ratio, threshold=threshold)
    model.cuda()
    model.eval()
    # flop, param = profile(model, inputs=(input,))

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None

    model.train()
    for x, _ in tqdm(train_loader):
        model(x.to("cuda"))
        
    model.eval()

    correct = 0
    total_num = 0
    total_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.cuda()
            y = y.cuda()
            logit = model(X)
            loss = F.cross_entropy(logit, y)
            correct += (logit.max(1)[1] == y).sum().item()
            total_num += y.size(0)
            total_loss += loss.item()
    print(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}")
    # print(
        # f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    with open(os.path.join(figure_path, "self_merge_result_mid.txt"), 'a+') as f:
        f.write(f"max_ratio:{max_ratio}, threshold:{threshold}\n")
        f.write(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}\n")
        # f.write(f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %\n\n")
    return model


def get_datasets(train=True, bs=512): #8
    path   = os.path.dirname(os.path.abspath(__file__))
    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        
    transform = transform_train
    if train is False:
        transform = transform_test

    mnistTrainSet = torchvision.datasets.CIFAR10(
        root=path + '/data', 
        train=train,
        download=True, 
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        mnistTrainSet,
        batch_size=bs, #256
        shuffle=True,
        num_workers=8)
    
    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_name", type=str, default="VGG11")
    parser.add_argument("--cluster_path", type=str, default=None)
    parser.add_argument("--distance_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--method", type=str, default="HDBSCAN", help="cluster method")
    parser.add_argument("--n_data_per_class", type=float, default=100)
    parser.add_argument("--min_sample", default=2, type=int)
    parser.add_argument("--max_ratio", default=0.5, type=float)
    parser.add_argument("--threshold", default=0.1, type=float)
    args = parser.parse_args()

    # load model
    vgg11_cfg       = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    features = make_layers(vgg11_cfg, batch_norm=True)
    model = VGG(features=features, num_classes=10)
    checkpoint = torch.load("/home/m/marza1/Iterative-Feature-Merging/vgg11_bn_1Xwider_CIFAR10.pt")

    model.load_state_dict(checkpoint)
    model.cuda()

    fuse_bnorms(model)
    return

    test_loader = get_datasets(train=False)
    train_loader = get_datasets(train=True)

    total_params = sum(p.numel() for p in model.parameters())
    new_model = test_activation_cluster(model, model.state_dict(), test_loader, train_loader, args.max_ratio, args.threshold, "./", args.vgg_name)
    new_total_params = sum(p.numel() for p in new_model.parameters())
    print("ACT SP", new_total_params / total_params)

if __name__ == "__main__":
  main()
