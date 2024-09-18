import torch.nn as nn
import torch.nn.functional as F
from utils.self_weight_matching import axes2perm_to_perm2axes, self_merge_weight_clustering
from functools import reduce


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu_conv = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu_conv(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_rep(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def get_axis_to_perm_ResNet50(approx_repair=False):
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,), f"{name}.running_mean": (p,),
                            f"{name}.running_var": (p,)}
    
    if approx_repair is True:
        conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,), f"{name}.bias": (p_out,)}

    dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}
    bottleneck = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"{name}.relu1"),
        **norm(f"{name}.bn1", f"{name}.relu1"),
        **conv(f"{name}.conv2", f"{name}.relu1", f"{name}.relu2"),
        **norm(f"{name}.bn2", f"{name}.relu2"),
        **conv(f"{name}.conv3", f"{name}.relu2", p_out),
        **norm(f"{name}.bn3", p_out),
    }
    axis_to_perm = {
        **conv('conv1', None, 'relu_conv'),
        **norm('bn1', 'relu_conv'),

        **bottleneck('layer1.0', 'relu_conv', 'layer1.0.relu3'),
        **conv('layer1.0.shortcut.0', 'relu_conv', 'layer1.0.relu3'),
        **norm('layer1.0.shortcut.1', 'layer1.0.relu3'),
        **bottleneck('layer1.1', 'layer1.0.relu3', 'layer1.0.relu3'),
        **bottleneck('layer1.2', 'layer1.0.relu3', 'layer1.0.relu3'),

        **bottleneck('layer2.0', 'layer1.0.relu3', 'layer2.0.relu3'),
        **conv('layer2.0.shortcut.0', "layer1.0.relu3", 'layer2.0.relu3'),
        **norm('layer2.0.shortcut.1', 'layer2.0.relu3'),
        **bottleneck('layer2.1', 'layer2.0.relu3', 'layer2.0.relu3'),
        **bottleneck('layer2.2', 'layer2.0.relu3', 'layer2.0.relu3'),
        **bottleneck('layer2.3', 'layer2.0.relu3', 'layer2.0.relu3'),

        **bottleneck('layer3.0', "layer2.0.relu3", 'layer3.0.relu3'),
        **conv('layer3.0.shortcut.0', "layer2.0.relu3", 'layer3.0.relu3'),
        **norm('layer3.0.shortcut.1', 'layer3.0.relu3'),
        **bottleneck('layer3.1', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.2', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.3', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.4', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.4', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.5', 'layer3.0.relu3', 'layer3.0.relu3'),

        **bottleneck('layer4.0', 'layer3.0.relu3', 'layer4.0.relu3'),
        **conv('layer4.0.shortcut.0', 'layer3.0.relu3', 'layer4.0.relu3'),
        **norm('layer4.0.shortcut.1', 'layer4.0.relu3'),
        **bottleneck('layer4.1', 'layer4.0.relu3', 'layer4.0.relu3'),
        **bottleneck('layer4.2', 'layer4.0.relu3', 'layer4.0.relu3'),

        **dense('linear', 'layer4.0.relu3', None)
    }

    return axis_to_perm

def get_axis_to_perm_ResNet18(approx_repair=False):
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}
    conv_b = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,), f"{name}.bias": (p_out,)}

    if approx_repair is True:
        conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,), f"{name}.bias": (p_out,)}

    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,), f"{name}.running_mean": (p,),
                            f"{name}.running_var": (p,)}
    
    dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}
    basicblock = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"{name}.relu1"),
        **norm(f"{name}.bn1", f"{name}.relu1"),
        **conv(f"{name}.conv2", f"{name}.relu1", p_out),
        **norm(f"{name}.bn2", p_out),
    }
    axis_to_perm = {
        **conv('conv1', None, 'relu_conv'),
        **norm('bn1', 'relu_conv'),

        **basicblock('layer1.0', 'relu_conv', 'relu_conv'),
        **basicblock('layer1.1', 'relu_conv', 'relu_conv'),

        **basicblock('layer2.0', 'relu_conv', 'layer2.0.relu2'),
        **conv('layer2.0.shortcut.0', 'relu_conv', 'layer2.0.relu2'),
        **norm('layer2.0.shortcut.1', 'layer2.0.relu2'),
        **basicblock('layer2.1', 'layer2.0.relu2', 'layer2.0.relu2'),

        **basicblock('layer3.0', 'layer2.0.relu2', 'layer3.0.relu2'),
        **conv('layer3.0.shortcut.0', 'layer2.0.relu2', 'layer3.0.relu2'),
        **norm('layer3.0.shortcut.1', 'layer3.0.relu2'),
        **basicblock('layer3.1', 'layer3.0.relu2', 'layer3.0.relu2'),

        **basicblock('layer4.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **conv('layer4.0.shortcut.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **norm('layer4.0.shortcut.1', 'layer4.0.relu2'),
        **basicblock('layer4.1', 'layer4.0.relu2', 'layer4.0.relu2'),

        **dense('linear', 'layer4.0.relu2', None)
    }

    return axis_to_perm


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    
    return reduce(getattr, names, module)


def merge_channel_ResNet50_clustering(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_ResNet50()
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()

    return origin_model


def merge_channel_ResNet50_clustering_approx_repair(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_ResNet50(approx_repair=True)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks, approx_repair=True)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()

    return origin_model


def merge_channel_ResNet18_clustering(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_ResNet18()
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model


def merge_channel_ResNet18_clustering_approx_repair(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_ResNet18(approx_repair=True)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks, approx_repair=True)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model
