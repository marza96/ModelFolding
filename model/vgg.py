import torch
import torch.nn as nn
from utils.self_weight_matching import axes2perm_to_perm2axes, self_merge_weight_clustering
from functools import reduce


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def get_axis_to_perm(model):
    next_perm = None
    axis_to_perm = {}
    max_cl_idx = 0
    min_cl_idx = 0
    max_conv_idx = 0
    min_conv_idx = 0

    for name, module in model.named_modules():
        if "classifier" in name and isinstance(module, nn.Linear):
            # print("calsss")
            max_cl_idx = max(max_cl_idx, int(name.split(".")[1]))
            min_cl_idx = min(min_cl_idx, int(name.split(".")[1]))
        if isinstance(module, nn.Conv2d):
            # print("name", name)
            max_conv_idx = max(max_conv_idx, int(name.split(".")[1]))
            min_conv_idx = min(min_conv_idx, int(name.split(".")[1]))

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            previous_perm = next_perm
            next_perm = f"perm_{name}"

            axis_to_perm[f"{name}.weight"] = (next_perm, previous_perm, None, None)
            axis_to_perm[f"{name}.bias"] = (next_perm, None)
        elif isinstance(module, nn.BatchNorm2d):
            axis_to_perm[f"{name}.weight"] = (next_perm, None)
            axis_to_perm[f"{name}.bias"] = (next_perm, None)
            axis_to_perm[f"{name}.running_mean"] = (next_perm, None)
            axis_to_perm[f"{name}.running_var"] = (next_perm, None)
            axis_to_perm[f"{name}.num_batches_tracked"] = ()
        elif isinstance(module, nn.Linear):
            previous_perm = next_perm
            next_perm = f"perm_{name}"

            if int(name.split(".")[1]) == max_cl_idx:
                next_perm = None

            axis_to_perm[f"{name}.weight"] = (next_perm, previous_perm)
            axis_to_perm[f"{name}.bias"] = (next_perm, )

    return axis_to_perm


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    
    return reduce(getattr, names, module)


def merge_channel_vgg11_clustering(origin_model, model_param, max_ratio=0.5, threshold=0.1):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm(origin_model)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=None)
    
    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model, perm_size


def merge_channel_vgg11_clustering_approx_repair(origin_model, model_param, max_ratio=0.5, threshold=0.1):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm(origin_model)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=None, approx_repair=True)
    
    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model, perm_size


def merge_channel_vgg_clustering(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm()
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()

    return origin_model

