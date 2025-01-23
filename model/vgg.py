import copy
import torch.nn as nn

from utils.weight_clustering import axes2perm_to_perm2axes, self_merge_weight_clustering, self_merge_weight_merging
from functools import reduce


class MergeVGG11ConvMlp:
    def __init__(self, key):
        self.key = key

    def requires(self, wk):
        return self.key in wk
    
    def merge(self, wk, p, merge):
        p  = p.reshape(4096, 512, 7*7)
        sh = p.shape
        p  = p.permute(1, 0, 2)
        p  = p.reshape(sh[1], -1)

        merged = merge @ p.clone().detach()
        merged = merged.reshape(merge.shape[0], 4096, 7 * 7)
        merged = merged.permute(1, 0, 2)
        merged = merged.reshape(4096, -1)

        return merged


def get_axis_to_perm(model):
    next_perm = None
    axis_to_perm = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            previous_perm = next_perm
            next_perm     = f"perm_{name}"

            axis_to_perm[f"{name}.weight"] = (next_perm, previous_perm, None, None)
            axis_to_perm[f"{name}.bias"]   = (next_perm, None)
        elif isinstance(module, nn.BatchNorm2d):
            axis_to_perm[f"{name}.weight"]              = (next_perm, None)
            axis_to_perm[f"{name}.bias"]                = (next_perm, None)
            axis_to_perm[f"{name}.running_mean"]        = (next_perm, None)
            axis_to_perm[f"{name}.running_var"]         = (next_perm, None)
            axis_to_perm[f"{name}.num_batches_tracked"] = ()
        elif isinstance(module, nn.Linear):
            previous_perm = next_perm
            next_perm     = f"perm_{name}"

            if "classifier.6" in name or ("layers.30" in name and isinstance(module, nn.Linear)):
                next_perm = None

            axis_to_perm[f"{name}.weight"] = (next_perm, previous_perm)
            axis_to_perm[f"{name}.bias"]   = (next_perm, )

    return axis_to_perm


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    
    return reduce(getattr, names, module)


def merge_channel_vgg11_clustering(origin_model, model_param, max_ratio=0.5, threshold=0.1):
    max_ratio    = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm(origin_model)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    
    custom_merger    = MergeVGG11ConvMlp("classifier.0")
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=None, custom_merger=custom_merger)
    
    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model, perm_size


def merge_channel_vgg11_clustering_approx_repair(origin_model, model_param, max_ratio=0.5, threshold=0.1):
    max_ratio    = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm(origin_model)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)

    custom_merger    = MergeVGG11ConvMlp("classifier.0")
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=None, approx_repair=True, custom_merger=custom_merger)
    
    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model, perm_size


def fuse_channel_vgg11_clustering(origin_model_a, origin_model_b, model_param_a, model_param_b):
    axes_to_perm = get_axis_to_perm(origin_model_a)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_merging(perm_to_axes, model_param_a, model_param_b, hooks=None)
    
    res_model = copy.deepcopy(origin_model_a)
    for p in param.keys():
        get_module_by_name(res_model, p).data = param[p].data.clone().detach()
    
    return res_model, perm_size


'''
    Custom VGG implementation (Smaller than torch's VGG)
'''
class VGG_small(nn.Module):
    def __init__(self, cfg, w=1, classes=10, in_channels=3, bnorm=False):
        super().__init__()

        self.in_channels = in_channels
        self.w           = w
        self.bnorm       = bnorm
        self.classes     = classes
        self.layers      = self._make_layers(cfg)

    def forward(self, x):
        out = self.layers[:-2](x)
        out = out.view(out.size(0), -1)
        out = self.layers[-2](out)
        out = self.layers[-1](out)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(nn.Conv2d(in_channels if in_channels == 3 else self.w*in_channels,
                                     self.w*x, kernel_size=3, padding=1))
                
                if self.bnorm is True:
                    layers.append(nn.BatchNorm2d(self.w*x))

                layers.append(nn.ReLU(inplace=True))
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Linear(self.w * cfg[-2], self.classes)]

        if self.bnorm is True:
            layers.append(nn.BatchNorm1d(self.classes))

        return nn.Sequential(*layers)

