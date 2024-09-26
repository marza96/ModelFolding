import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.self_weight_matching import axes2perm_to_perm2axes, self_merge_weight_clustering
from typing import Any, Callable, List, Optional, Type, Union

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
    

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck_Wider(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet_Wider(nn.Module):
    def __init__(
        self,
        block: Type[Union[Bottleneck_Wider]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        wider_factor =1,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        
        self.groups = groups
        self.base_width = width_per_group * wider_factor
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[Bottleneck_Wider]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def ResNet50Wider(num_classes=10, width_factor=1):
    return ResNet_Wider(Bottleneck_Wider, [3, 4, 6, 3], num_classes=num_classes, wider_factor=width_factor)

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


def get_axis_to_perm_ResNet50(approx_repair=False, override=True):
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,), f"{name}.running_mean": (p,),
                            f"{name}.running_var": (p,)}
    
    if approx_repair is True:
        conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,), f"{name}.bias": (p_out,)}

    sh = "shortcut"
    fc = "linear"
    if override is False:
        sh = "downsample"
        fc = "fc"

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
        **conv(f'layer1.0.{sh}.0', 'relu_conv', 'layer1.0.relu3'),
        **norm(f'layer1.0.{sh}.1', 'layer1.0.relu3'),
        **bottleneck('layer1.1', 'layer1.0.relu3', 'layer1.0.relu3'),
        **bottleneck('layer1.2', 'layer1.0.relu3', 'layer1.0.relu3'),

        **bottleneck('layer2.0', 'layer1.0.relu3', 'layer2.0.relu3'),
        **conv(f'layer2.0.{sh}.0', "layer1.0.relu3", 'layer2.0.relu3'),
        **norm(f'layer2.0.{sh}.1', 'layer2.0.relu3'),
        **bottleneck('layer2.1', 'layer2.0.relu3', 'layer2.0.relu3'),
        **bottleneck('layer2.2', 'layer2.0.relu3', 'layer2.0.relu3'),
        **bottleneck('layer2.3', 'layer2.0.relu3', 'layer2.0.relu3'),

        **bottleneck('layer3.0', "layer2.0.relu3", 'layer3.0.relu3'),
        **conv(f'layer3.0.{sh}.0', "layer2.0.relu3", 'layer3.0.relu3'),
        **norm(f'layer3.0.{sh}.1', 'layer3.0.relu3'),
        **bottleneck('layer3.1', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.2', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.3', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.4', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.4', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.5', 'layer3.0.relu3', 'layer3.0.relu3'),

        **bottleneck('layer4.0', 'layer3.0.relu3', 'layer4.0.relu3'),
        **conv(f'layer4.0.{sh}.0', 'layer3.0.relu3', 'layer4.0.relu3'),
        **norm(f'layer4.0.{sh}.1', 'layer4.0.relu3'),
        **bottleneck('layer4.1', 'layer4.0.relu3', 'layer4.0.relu3'),
        **bottleneck('layer4.2', 'layer4.0.relu3', 'layer4.0.relu3'),

        **dense(f'{fc}', 'layer4.0.relu3', None)
    }

    return axis_to_perm

def get_axis_to_perm_ResNet18(approx_repair=False, override=True):
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}
    conv_b = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,), f"{name}.bias": (p_out,)}

    if approx_repair is True:
        conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,), f"{name}.bias": (p_out,)}

    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,), f"{name}.running_mean": (p,),
                            f"{name}.running_var": (p,)}
    
    sh = "shortcut"
    fc = "linear"
    if override is False:
        sh = "downsample"
        fc = "fc"

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
        **conv(f'layer2.0.{sh}.0', 'relu_conv', 'layer2.0.relu2'),
        **norm(f'layer2.0.{sh}.1', 'layer2.0.relu2'),
        **basicblock('layer2.1', 'layer2.0.relu2', 'layer2.0.relu2'),

        **basicblock('layer3.0', 'layer2.0.relu2', 'layer3.0.relu2'),
        **conv(f'layer3.0.{sh}.0', 'layer2.0.relu2', 'layer3.0.relu2'),
        **norm(f'layer3.0.{sh}.1', 'layer3.0.relu2'),
        **basicblock('layer3.1', 'layer3.0.relu2', 'layer3.0.relu2'),

        **basicblock('layer4.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **conv(f'layer4.0.{sh}.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **norm(f'layer4.0.{sh}.1', 'layer4.0.relu2'),
        **basicblock('layer4.1', 'layer4.0.relu2', 'layer4.0.relu2'),

        **dense(f'{fc}', 'layer4.0.relu2', None)
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


def merge_channel_ResNet50_clustering_wider(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_ResNet50(override=False)
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


def merge_channel_ResNet50_clustering_approx_repair_wider(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_ResNet50(approx_repair=True, override=False)
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


def merge_channel_ResNet18_big_clustering(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_ResNet18(override=False)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model



def merge_channel_ResNet50_big_clustering(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_ResNet50(override=False)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model


def merge_channel_ResNet18_big_clustering_approx_repair(origin_model, model_param, max_ratio=1., threshold=0.1, hooks=None):
    max_ratio = 1.0 - max_ratio
    axes_to_perm = get_axis_to_perm_ResNet18(override=False, approx_repair=True)
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight_clustering(perm_to_axes, model_param, max_ratio, threshold, hooks=hooks, approx_repair=True)

    for p in param.keys():
        get_module_by_name(origin_model, p).data = param[p].data.clone().detach()
    
    return origin_model
