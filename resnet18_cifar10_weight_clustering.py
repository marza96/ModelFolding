from model.resnet import ResNet18, merge_channel_ResNet18_clustering
from utils.utils import eval_model, load_model, AvgLayerStatisticsHook
from utils.utils import DI_REPAIR, NO_REPAIR, REPAIR
from utils.datasets import get_cifar10
from thop import profile
from tqdm import tqdm

import wandb
import argparse
import torch
import copy


def measure_avg_var(model, dataloader):
    hooks = {
        "bn1": AvgLayerStatisticsHook(conv=True),
        "relu_conv": AvgLayerStatisticsHook(conv=True),
        "layer1.0.relu1": AvgLayerStatisticsHook(conv=True),
        "layer1.1.relu1": AvgLayerStatisticsHook(conv=True),
        "layer2.0.relu1": AvgLayerStatisticsHook(conv=True),
        "layer2.0.relu2": AvgLayerStatisticsHook(conv=True),
        "layer2.1.relu1": AvgLayerStatisticsHook(conv=True),
        "layer3.0.relu1": AvgLayerStatisticsHook(conv=True),
        "layer3.0.relu2": AvgLayerStatisticsHook(conv=True),
        "layer3.1.relu1": AvgLayerStatisticsHook(conv=True),
        "layer4.0.relu1": AvgLayerStatisticsHook(conv=True),
        "layer4.0.relu2": AvgLayerStatisticsHook(conv=True),
        "layer4.1.relu1": AvgLayerStatisticsHook(conv=True)
    }

    handles = list()
    handles.append(model.conv1.register_forward_hook(hooks["bn1"]))
    handles.append(model.conv1.register_forward_hook(hooks["relu_conv"]))
    handles.append(model.layer1[0].conv1.register_forward_hook(hooks["layer1.0.relu1"]))
    handles.append(model.layer1[1].conv1.register_forward_hook(hooks["layer1.1.relu1"]))
    handles.append(model.layer2[0].conv1.register_forward_hook(hooks["layer2.0.relu1"]))
    handles.append(model.layer2[0].conv2.register_forward_hook(hooks["layer2.0.relu2"]))
    handles.append(model.layer2[1].conv1.register_forward_hook(hooks["layer2.1.relu1"]))
    handles.append(model.layer3[0].conv1.register_forward_hook(hooks["layer3.0.relu1"]))
    handles.append(model.layer3[0].conv2.register_forward_hook(hooks["layer3.0.relu2"]))
    handles.append(model.layer3[1].conv1.register_forward_hook(hooks["layer3.1.relu1"]))
    handles.append(model.layer4[0].conv1.register_forward_hook(hooks["layer4.0.relu1"]))
    handles.append(model.layer4[0].conv2.register_forward_hook(hooks["layer4.0.relu2"]))
    handles.append(model.layer4[1].conv1.register_forward_hook(hooks["layer4.1.relu1"]))

    model.cuda()
    model.eval()
 
    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(tqdm(dataloader)), desc="Eval initial stats"):
            X = X.cuda()
            y = y.cuda()
            _ = model(X)

    avg_vars = list()
    for key in hooks.keys():
        avg_vars.append(hooks[key].get_stats())

    for handle in handles:
        handle.remove()

    return avg_vars


def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, method, repair, di_samples_path, eval=True, measure_variance=False):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    var = None
    origin_var = None

    if measure_variance is True:
        origin_var = measure_avg_var(origin_model, dataloader)

    origin_flop, origin_param = profile(origin_model, inputs=(input,))

    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, hooks=None)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if repair != NO_REPAIR:
        if repair == DI_REPAIR: 
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            model(torch.load(di_samples_path).to("cuda"))
            model.eval()

        elif repair == REPAIR:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            for x, _ in tqdm(train_loader):
                model(x.to("cuda"))
                
            model.eval()
    
    if measure_variance is True:
        var = measure_avg_var(model, dataloader)
        ratios = list()
        for i in range(len(var)):
            ratios.append(float((var[i] / origin_var[i])))

    if eval is True:
        acc, loss = eval_model(model, dataloader)
        print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")

        print(
            f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    
        if var is not None:
            return model, acc, param / origin_param, float((var[-1] / origin_var[-1]))
            
        return model, acc, param / origin_param, -1

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--repair", type=str, default="NO_REPAIR", help="")
    parser.add_argument("--proj_name", type=str, help="", default="Folding Resnet18 cifar10")
    parser.add_argument("--exp_name", type=str, help="", default="WM REPAIR")
    parser.add_argument("--di_samples_path", type=str, default="cifar10.pt")
    args = parser.parse_args()

    model = ResNet18()
    load_model(model, args.checkpoint)
    model.cuda()

    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)

    desc = {"experiment": args.exp_name}
    wandb.init(
        project=args.proj_name,
        config=desc,
        name=args.exp_name
    )
    for ratio in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity, var_ratio = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, merge_channel_ResNet18_clustering, args.repair, args.di_samples_path)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": 1.0 - sparsity})
        wandb.log({"var_ratio": var_ratio})


if __name__ == "__main__":
  main()
