
from model.resnet import merge_channel_ResNet18_big_clustering
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm
from utils.utils import load_model
from PIL import ImageFile
from thop import profile

import copy
import torch
import wandb
import torch
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_imagenet(datadir, train=True, bs=256):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    get_dataset = getattr(datasets, "ImageNet")

    normalize = transforms.Normalize(mean=mean, std=std)
    tr_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    if train is True:
        dataset = get_dataset(root=datadir, split='train', transform=tr_transform)
    else:
        dataset = get_dataset(root=datadir, split='val', transform=val_transform)

    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,num_workers=8)

    return data_loader


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, figure_path, method, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, threshold=threshold, hooks=None)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if eval is True:
        if eval is True:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            for x, _ in tqdm(train_loader):
                model(x.to("cuda"))
                break
                
            model.eval()

        # for module in model.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         module.reset_running_stats()
        #         module.momentum = None

        # model.train()
        # print("DI")
        # model(torch.load("imagenet_di.pt").to("cuda"))
        # model.eval()


    if eval is True:
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

    print(
        f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    # with open(os.path.join(figure_path, "self_merge_result.txt"), 'a+') as f:
    #     f.write(f"max_ratio:{max_ratio}, threshold:{threshold}\n")
    #     f.write(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}\n")
    #     f.write(f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %\n\n")
    
    if eval is True:
        return model, correct/total_num, param / origin_param

    return model


def main():
    test_loader = get_imagenet("/home/m/marza1/imagenet/ImageNet/ILSVRC12", train=False, bs=512)
    train_loader = get_imagenet("/home/m/marza1/imagenet/ImageNet/ILSVRC12", train=True, bs=256)

    model = resnet18(num_classes=1000).to("cuda")
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet18_imagenet.pt", override=False)

    proj_name = "CLUSTERING Imagenet New"

    exp_name = "Weight-Clustering REPAIR"
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

    correct = 0
    total_num = 0
    total_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(test_loader)):
            X = X.cuda()
            y = y.cuda()
            logit = model(X)
            loss = F.cross_entropy(logit, y)
            correct += (logit.max(1)[1] == y).sum().item()
            total_num += y.size(0)
            total_loss += loss.item()
    print(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}")


if __name__ == "__main__":
    main()