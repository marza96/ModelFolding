from model.resnet import ResNet18, merge_channel_ResNet18, merge_channel_ResNet18_clustering

import wandb
import argparse
import torch
import os
import copy
from torchvision import datasets, transforms

from tqdm import tqdm
import torchvision

from thop import profile

import torch.nn.functional as F

def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, threshold, figure_path, method, eval=True):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    
    hooks = None
    # hooks = {
    #     "bn1": LayerActivationHook(),
    #     "relu_conv": LayerActivationHook(),
    #     "layer1.0.relu1": LayerActivationHook(),
    #     "layer1.1.relu1": LayerActivationHook(),
    #     "layer2.0.relu1": LayerActivationHook(),
    #     "layer2.0.relu2": LayerActivationHook(),
    #     "layer2.1.relu1": LayerActivationHook(),
    #     "layer3.0.relu1": LayerActivationHook(),
    #     "layer3.0.relu2": LayerActivationHook(),
    #     "layer3.1.relu1": LayerActivationHook(),
    #     "layer4.0.relu1": LayerActivationHook(),
    #     "layer4.0.relu2": LayerActivationHook(),
    #     "layer4.1.relu1": LayerActivationHook()
    # }

    # handles = list()
    # handles.append(origin_model.bn1.register_forward_hook(hooks["bn1"]))
    # handles.append(origin_model.conv1.register_forward_hook(hooks["relu_conv"]))
    # handles.append(origin_model.layer1[0].conv1.register_forward_hook(hooks["layer1.0.relu1"]))
    # handles.append(origin_model.layer1[1].conv1.register_forward_hook(hooks["layer1.1.relu1"]))
    # handles.append(origin_model.layer2[0].conv1.register_forward_hook(hooks["layer2.0.relu1"]))
    # handles.append(origin_model.layer2[0].conv2.register_forward_hook(hooks["layer2.0.relu2"]))
    # handles.append(origin_model.layer2[1].conv1.register_forward_hook(hooks["layer2.1.relu1"]))
    # handles.append(origin_model.layer3[0].conv1.register_forward_hook(hooks["layer3.0.relu1"]))
    # handles.append(origin_model.layer3[0].conv2.register_forward_hook(hooks["layer3.0.relu2"]))
    # handles.append(origin_model.layer3[1].conv1.register_forward_hook(hooks["layer3.1.relu1"]))
    # handles.append(origin_model.layer4[0].conv1.register_forward_hook(hooks["layer4.0.relu1"]))
    # handles.append(origin_model.layer4[0].conv2.register_forward_hook(hooks["layer4.0.relu2"]))
    # handles.append(origin_model.layer4[1].conv1.register_forward_hook(hooks["layer4.1.relu1"]))

    # origin_model(torch.load("input_tens.pt").to("cuda"))

    # for handle in handles:
    #     handle.remove()

    # with torch.no_grad():
    #     for x, _ in tqdm(train_loader):
    #         origin_model(x.to("cuda"))
    #         break

    # # rm, rv = hooks["bn1"].get_stats()
    # # mu, std = origin_model.bn1.bias, origin_model.bn1.weight
    # # print(rm)
    # # print(mu)

    # mu, std = origin_model.bn1.bias, origin_model.bn1.weight

    # rm_pred, rv_pred = predict_relu_stats(mu , std)
    # rm, rv = hooks["relu_conv"].get_stats()

    

    # mus = mu.clone().detach()
    # stds = std.clone().detach()
    # for i in range(8):
    #     mus = torch.vstack((mus, mus))
    #     stds = torch.vstack((stds, stds))

    # in_samples = torch.normal(mus, stds)
    # out_samples = F.relu(in_samples)
    # # rv_pred = out_samples.var(dim=0)

    # relu = nn.ReLU()
    # hook = LayerStatisticsHook(conv=False)
    # relu.register_forward_hook(hook)
    # for i in range(50):
    #     in_samples = torch.normal(mus, stds)
    #     _ = relu(in_samples)
    # rm_pred, rv_pred = hook.get_stats()

    # print(rm)
    # print(rm_pred)
    
    # return None
    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, threshold=threshold, hooks=hooks)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if eval is True:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = None

        model.train()
        for x, _ in tqdm(train_loader):
            model(x.to("cuda"))
            
        model.eval()

    # if eval is True:
    #     for module in model.modules():
    #         if isinstance(module, torch.nn.BatchNorm2d):
    #             module.reset_running_stats()
    #             module.momentum = None

    #     model.train()
    #     model(torch.load("input_tens.pt").to("cuda"))
    #     model.eval()

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


# def test_merge(origin_model, checkpoint, dataloader, max_ratio, threshold, figure_path, method):
#     input = torch.torch.randn(1, 3, 32, 32).cuda()
#     origin_model.cuda()
#     origin_model.eval()
#     origin_flop, origin_param = profile(origin_model, inputs=(input,))
#     model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, threshold=threshold)
#     model.cuda()
#     model.eval()
#     flop, param = profile(model, inputs=(input,))

#     # for module in model.modules():
#     #     if isinstance(module, torch.nn.BatchNorm2d):
#     #         module.reset_running_stats()

#     # model.train()
#     # for x, _ in tqdm(dataloader):
#     #     model(x.to("cuda"))
#     # model.eval()

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
#     print(
#         f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
#     with open(os.path.join(figure_path, "self_merge_result.txt"), 'a+') as f:
#         f.write(f"max_ratio:{max_ratio}, threshold:{threshold}\n")
#         f.write(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}\n")
#         f.write(f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %\n\n")
    
#     return model, correct/total_num, param / origin_param


def load_model(model, i):
    sd = torch.load(i, map_location=torch.device('cpu'))
    new_sd = copy.deepcopy(sd)

    for key, value in sd.items():
        if "downsample" in key:
            new_key = key.replace("downsample", "shortcut")
            new_sd[new_key] = value
            new_sd.pop(key)

        if "fc" in key:
            new_key = key.replace("fc", "linear")
            new_sd[new_key] = value
            new_sd.pop(key)

    model.load_state_dict(new_sd)


def get_datasets(train=True, bs=512): #8
    path   = os.path.dirname(os.path.abspath(__file__))
    
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    transform = transform_train
    if train is False:
        transform = transform_test

    mnistTrainSet = torchvision.datasets.CIFAR100(
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
    proj_name = "resnet18 CIFAR100"
    
    model = ResNet18(num_classes=100)
    load_model(model, "/home/m/marza1/Iterative-Feature-Merging/resnet18_1Xwider_CIFAR100.pt")
    model.cuda()

    test_loader = get_datasets(train=False)
    train_loader = get_datasets(train=True)

    max_ratio=0.5
    threshold=10.95
    figure_path = '/home/m/marza1/Iterative-Feature-Merging/'
    
    # ratio = max_ratio
    # total_params = sum(p.numel() for p in model.parameters())
    # new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, threshold, figure_path, merge_channel_ResNet18_clustering)
    # new_total_params = sum(p.numel() for p in new_model.parameters())
    # print("ACT SP", new_total_params / total_params)

    exp_name = "Weight-Clustering"
    desc = {"experiment": exp_name}
    wandb.init(
        project=proj_name,
        config=desc,
        name=exp_name
    )
    for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, ratio, threshold, figure_path, merge_channel_ResNet18_clustering)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": sparsity})

    # exp_name = "IFM"
    # desc = {"experiment": exp_name}
    # wandb.init(
    #     project=proj_name,
    #     config=desc,
    #     name=exp_name,
    #     reinit=True
    # )
    # for ratio in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
    #     new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, ratio, threshold, figure_path, merge_channel_ResNet18)
    #     wandb.log({"test acc": acc})
    #     wandb.log({"sparsity": sparsity})

if __name__ == "__main__":
  main()
