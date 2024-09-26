from torchvision import transforms
from utils.pruning import local_structured_prune_model, reset_bn_stats
from utils.utils import load_model, eval
from utils.datasets import get_cifar10
from torchvision.models.vgg import VGG, make_layers


import torch
import wandb


def main():
    vgg11_cfg  = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    features   = make_layers(vgg11_cfg, batch_norm=True)
    model      = VGG(features=features, num_classes=10)
    checkpoint = torch.load("/home/m/marza1/Iterative-Feature-Merging/vgg11_bn_1Xwider_CIFAR10.pt", map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint)
    model.cuda()

    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)

    proj_name = "Folding VGG11 cifar10"
    exp_name = "L1 REPAIR"
    desc = {"experiment": exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=exp_name
    )

    # model = ResNet18()
    # load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet18_1Xwider_CIFAR10_latest.pt")
    # model.cuda()
    # local_structured_prune_model(model, 0.2, n=1, save_path=None)  
    # # reset_bn_stats(model, train_loader, "cuda")
    # acc = eval(model, test_loader)
    # print("ACC", acc)
    
    for sp in [0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.95, 0.97]:
        local_structured_prune_model(model, sp, n=1, save_path=None)  
        reset_bn_stats(model, train_loader, "cuda")
        acc = eval(model, test_loader)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sp})
    

if __name__ == "__main__":
    main()