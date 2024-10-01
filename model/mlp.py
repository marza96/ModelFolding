import torch

from torch import nn
from typing import Any
from functools import wraps, reduce
from utils.weight_clustering import axes2perm_to_perm2axes, self_merge_weight_clustering

__all__ = ['MLP', 'mlp']


def clean_mlp_kwargs(func):
    @wraps(func)
    def wrapper(**kwargs):
        valid_kwargs = ['num_classes', 'wider_factor', 'n_channels', 'weights']
        extra_kwargs = {k: kwargs[k] for k in kwargs if k not in valid_kwargs}
        if extra_kwargs:
            print(f"Removed unsupported kwargs: {list(extra_kwargs.keys())}")
            kwargs = {k: kwargs[k] for k in kwargs if k in valid_kwargs}
        return func(**kwargs)
    return wrapper


class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, wider_factor: int = 1, n_channels: int = 3):
        super(MLP, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.fc1 = nn.Linear(n_channels*32*32, 512*wider_factor)
        self.bn1 = nn.BatchNorm1d(512*wider_factor)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512*wider_factor, 256*wider_factor)
        self.bn2 = nn.BatchNorm1d(256*wider_factor)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256*wider_factor, 128*wider_factor)
        self.bn3 = nn.BatchNorm1d(128*wider_factor)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128*wider_factor, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 32*32*self.n_channels)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


@clean_mlp_kwargs
def mlp(**kwargs: Any) -> MLP:
    pre_kwargs = {k: kwargs[k] for k in kwargs if k != 'weights'}
    model = MLP(**pre_kwargs)
    print("Create mlp: initialized")
    if 'weights' in kwargs:
        model.load_state_dict(torch.load(kwargs['weights'], map_location='cpu'))
        print(f"Load mlp weights from {kwargs['weights']}")
    return model


def get_axis_to_perm_mlp(approx_repair=False, override=True):
    return {
        "fc1.weight": ("fc1", None),
        "fc1.bias": ("fc1", ),
        "bn1.weight": ("fc1",), 
        "bn1.bias": ("fc1",), 
        "bn1.running_mean": ("fc1",),                    
        "bn1.running_var": ("fc1",),

        "fc2.weight": ("fc2", "fc1"),
        "fc2.bias": ("fc2", ),
        "bn2.weight": ("fc2",), 
        "bn2.bias": ("fc2",), 
        "bn2.running_mean": ("fc2",),                    
        "bn2.running_var": ("fc2",),

        "fc3.weight": ("fc3", "fc2"),
        "fc3.bias": ("fc3", ),
        "bn3.weight": ("fc3",), 
        "bn3.bias": ("fc3",), 
        "bn3.running_mean": ("fc3",),                    
        "bn3.running_var": ("fc3",),

        "fc4.weight": (None, "fc3"),
    }


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    
    return reduce(getattr, names, module)


def merge_channel_mlp_clustering(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_mlp()
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model


def merge_channel_mlp_clustering_approx_repair(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_mlp(approx_repair=True)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks, approx_repair=True)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model