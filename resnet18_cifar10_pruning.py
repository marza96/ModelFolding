from torchvision import transforms
from utils.pruning import local_structured_prune_model, reset_bn_stats
from utils.utils import load_model, eval
from model.resnet import ResNet18

import os
import torch
import wandb
import torchvision


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
        batch_size=bs, 
        shuffle=True,
        num_workers=8)
    
    return loader


def main():
    model = ResNet18()
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet18_1Xwider_CIFAR10_latest.pt")
    model.cuda()

    test_loader = get_datasets(train=False)
    train_loader = get_datasets(train=True)

    # proj_name = "structured pruning"
    # exp_name = "L2 REPAIR"
    # desc = {"experiment": exp_name}
    # wandb.init(
    #     project=proj_name,
    #     config=desc,
    #     name=exp_name
    # )

    model = ResNet18()
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet18_1Xwider_CIFAR10_latest.pt")
    model.cuda()
    local_structured_prune_model(model, 0.2, n=1, save_path=None)  
    # reset_bn_stats(model, train_loader, "cuda")
    acc = eval(model, test_loader)
    print("ACC", acc)
    
    # for sp in [0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.95, 0.97]:
    #     model = ResNet18()
    #     load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet18_1Xwider_CIFAR10_latest.pt")
    #     model.cuda()
    #     local_structured_prune_model(model, sp, n=2, save_path=None)  
    #     reset_bn_stats(model, train_loader, "cuda")
    #     acc = eval(model, test_loader)
    #     wandb.log({"test acc": acc})
    #     wandb.log({"sparsity": sp})
    

if __name__ == "__main__":
    main()