# from model.vgg import merge_channel_vgg11_clustering, get_axis_to_perm
# import argparse
# import torch
# import wandb
# import copy
# from tqdm import tqdm
# from thop import profile
# import torch.nn.functional as F
# from torchvision.models.vgg import VGG, make_layers
# from utils.datasets import get_imagenet


# def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, method):
#     input = torch.torch.randn(1, 3, 32, 32).cuda()
#     origin_model.cuda()
#     origin_model.eval()

#     origin_flop, origin_param = profile(origin_model, inputs=(input,))
#     model, _ = method(origin_model, checkpoint, max_ratio=max_ratio, threshold=threshold)
#     model.cuda()
#     model.eval()
#     flop, param = profile(model, inputs=(input,))
    
#     for module in model.modules():
#         if isinstance(module, torch.nn.BatchNorm2d):
#             module.reset_running_stats()
#             module.momentum = None

#     data = torch.load("imagenet_di.pt", map_location="cpu")
#     model.train()
#     print("DI")
#     model(data[:128, :, :, :].cuda())
#     model(data[128:, :, :, :].cuda())
#     model.eval()

#     correct = 0
#     total_num = 0
#     total_loss = 0
#     with torch.no_grad():
#         for i, (X, y) in enumerate(tqdm(dataloader)):
#             X = X.cuda()
#             y = y.cuda()
#             logit = model(X)
#             loss = F.cross_entropy(logit, y)
#             correct += (logit.max(1)[1] == y).sum().item()
#             total_num += y.size(0)
#             total_loss += loss.item()
#     print(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}")

#     return correct / total_num, param / origin_param


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--vgg_name", type=str, default="VGG11")
#     parser.add_argument("--cluster_path", type=str, default=None)
#     parser.add_argument("--distance_path", type=str, default=None)
#     parser.add_argument("--seed", type=int, default=0, help="Random seed")
#     parser.add_argument("--method", type=str, default="HDBSCAN", help="cluster method")
#     parser.add_argument("--n_data_per_class", type=float, default=100)
#     parser.add_argument("--min_sample", default=2, type=int)
#     parser.add_argument("--max_ratio", default=0.5, type=float)
#     parser.add_argument("--threshold", default=0.1, type=float)
#     args = parser.parse_args()

#     # load model
#     vgg11_cfg  = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
#     features   = make_layers(vgg11_cfg, batch_norm=True)
#     model      = VGG(features=features, num_classes=1000)
#     checkpoint = torch.load("/home/m/marza1/Iterative-Feature-Merging/vgg11_imagenet.pt")

#     model.load_state_dict(checkpoint)
#     model.cuda()

#     test_loader = get_imagenet("/home/m/marza1/imagenet/ImageNet/ILSVRC12", train=False, bs=512)
#     train_loader = get_imagenet("/home/m/marza1/imagenet/ImageNet/ILSVRC12", train=True, bs=256)

#     proj_name = "Folding VGG11 imagenet"

#     exp_name = "WM REPAIR"
#     desc = {"experiment": exp_name}
#     wandb.init(
#         project=proj_name,
#         config=desc,
#         name=exp_name,
#         reinit=True
#     )
#     for ratio in [0.025, 0.05, 0.1, 0.12, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
#         acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, 100.0,  merge_channel_vgg11_clustering)
#         print("ACC", acc)
#         wandb.log({"test acc": acc})
#         wandb.log({"sparsity": sparsity})


# if __name__ == "__main__":
#   main()


from model.vgg import merge_channel_vgg11_clustering, get_axis_to_perm
import argparse
import torch
import wandb
import copy
from tqdm import tqdm
from thop import profile
import torch.nn.functional as F
from torchvision.models.vgg import VGG, make_layers
from utils.datasets import get_imagenet
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, method):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    model, _ = method(origin_model, checkpoint, max_ratio=max_ratio, threshold=threshold)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))
    
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None

    data = torch.load("di_vgg_imagenet_256.pt", map_location="cpu")
    model.train()
    print("DI")
    model(data[:128, :, :, :].cuda())
    model(data[128:, :, :, :].cuda())
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

    return correct / total_num, param / origin_param


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
    vgg11_cfg  = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    features   = make_layers(vgg11_cfg, batch_norm=True)
    model      = VGG(features=features, num_classes=1000)
    checkpoint = torch.load("/home/m/marza1/Iterative-Feature-Merging/vgg11_imagenet.pt")

    model.load_state_dict(checkpoint)
    model.cuda()

    test_loader = get_imagenet("/home/m/marza1/imagenet/ImageNet/ILSVRC12", train=False, bs=64)
    train_loader = get_imagenet("/home/m/marza1/imagenet/ImageNet/ILSVRC12", train=True, bs=64)

    proj_name = "Folding VGG11 imagenet"

    exp_name = "WM DI REPAIR"
    desc = {"experiment": exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=exp_name,
        reinit=True
    )
    for ratio in [0.025, 0.05, 0.1, 0.12, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, 100.0,  merge_channel_vgg11_clustering)
        print("ACC", acc)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})


if __name__ == "__main__":
  main()
