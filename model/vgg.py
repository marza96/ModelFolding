import torch
import torch.nn as nn
from utils.self_weight_matching import axes2perm_to_perm2axes, self_merge_weight, self_merge_weight_clustering
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
            print("calsss")
            max_cl_idx = max(max_cl_idx, int(name.split(".")[1]))
            min_cl_idx = min(min_cl_idx, int(name.split(".")[1]))
        if isinstance(module, nn.Conv2d):
            print("name", name)
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

    
    for key in axis_to_perm.keys():
        print(key, axis_to_perm[key])

    return axis_to_perm


class VGG(nn.Module):
    def __init__(self, vgg_name, nclass=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(cfg[vgg_name][-2], nclass)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    @ staticmethod
    def get_axis_to_perm(model):
        next_perm = None
        axis_to_perm = {}
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
                axis_to_perm[f"{name}.weight"] = (None, next_perm)
                axis_to_perm[f"{name}.bias"] = (None, )
        return axis_to_perm


class VGG_cluster(nn.Module):
    def __init__(self, vgg_name, channel_num):
        super(VGG_cluster, self).__init__()
        self.channel_num = channel_num
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(channel_num[-1], 10)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        i = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.channel_num[i] is None:
                    n_c = x
                else:
                    n_c = self.channel_num[i]
                layers += [nn.Conv2d(in_channels, n_c, kernel_size=3, padding=1),
                           nn.BatchNorm2d(n_c),
                           nn.ReLU(inplace=True)]
                in_channels = n_c
                i += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_no_act(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_no_act, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.Identity(inplace=True)
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    @ staticmethod
    def get_axis_to_perm(model):
        next_perm = None
        axis_to_perm = {}
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
                axis_to_perm[f"{name}.weight"] = (None, next_perm)
                axis_to_perm[f"{name}.bias"] = (None)
        return axis_to_perm

def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    
    return reduce(getattr, names, module)


def merge_channel_vgg16(origin_model, vgg_name, model_param, max_ratio=0.5, threshold=0.1):
    axes_to_perm = get_axis_to_perm(origin_model)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    # param, perm_size = self_merge_weight(perm_to_axes, model_param, max_ratio, threshold)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=None)
    
    perms = perm_size.keys()

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


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

if __name__ == "__main__":
    model_names = ['VGG11']
    for model_name in model_names:
        model = VGG(model_name)
        layers_to_hook = []
        get_hook = False
        for name, module in model.named_modules():
            if get_hook:
                layers_to_hook.append(name)
                get_hook = False
            if isinstance(module, nn.Conv2d):
                mtype = "conv2d"
            elif isinstance(module, nn.BatchNorm2d):
                mtype = "bn2d"
            elif isinstance(module, nn.ReLU):
                mtype = "relu"
                get_hook = True
            elif isinstance(module, nn.MaxPool2d):
                mtype = "maxpool"
            else:
                mtype = "Unknown"
            # print(f"{name}:{mtype}")
        print(f"{model_name}: {layers_to_hook}")
