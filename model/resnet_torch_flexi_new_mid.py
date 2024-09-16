from torchvision.models import resnet50
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from utils.self_weight_matching import axes2perm_to_perm2axes, self_merge_weight


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

class BasicBlock(nn.Module):
    expansion: int = 1

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
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

class Bottleneck(nn.Module):
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
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
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
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu_conv = nn.ReLU(inplace=True)
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
        block: Type[Union[BasicBlock, Bottleneck]],
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

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu_conv(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def Wide_ResNet50_2():
    return ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=128)

class BasicBlock_flexi(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        width: list,
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
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if downsample is not None and width is not None:
            if width[0] is None:
                self.conv1 = conv3x3(inplanes, width[1], stride)
            else:
                self.conv1 = conv3x3(width[0], width[1], stride)
            self.bn1 = norm_layer(width[1])
            if width[2] is None:
                self.conv2 = conv3x3(width[1], planes)
                self.bn2 = norm_layer(planes)
            else:
                self.conv2 = conv3x3(width[1], width[2])
                self.bn2 = norm_layer(width[2])
        elif width is not None:
            self.conv1 = conv3x3(inplanes, width[1], stride)
            self.bn1 = norm_layer(width[1])
            self.conv2 = conv3x3(width[1], inplanes)
            self.bn2 = norm_layer(inplanes)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

class Bottleneck_flexi(nn.Module):
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
        width: list = None,
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
        #width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if downsample is not None and width is not None:
            if width[0] is None:
                self.conv1 = conv1x1(inplanes, width[1])
            else:
                self.conv1 = conv1x1(width[0], width[1])
            self.bn1 = norm_layer(width[1])
            self.conv2 = conv3x3(width[1], width[2], stride, groups, dilation)
            self.bn2 = norm_layer(width[2])
            if width[3] is None:
                self.conv3 = conv1x1(width[2], inplanes)
                self.bn3 = norm_layer(inplanes)
            else:
                self.conv3 = conv1x1(width[2], width[3])
                self.bn3 = norm_layer(width[3])
        elif width is not None:
            self.conv1 = conv1x1(inplanes, width[1])
            self.bn1 = norm_layer(width[1])
            self.conv2 = conv3x3(width[1], width[2], stride, groups, dilation)
            self.bn2 = norm_layer(width[2])
            self.conv3 = conv1x1(width[2], inplanes)
            self.bn3 = norm_layer(inplanes)
        else:
            width = int(planes * (base_width / 64.0)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class ResNet_flexi(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock_flexi, Bottleneck_flexi]],
        layers: List[int],
        width_list: List[List],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
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
        self.base_width = width_per_group
        if width_list[0] is not None:
            self.conv1 = nn.Conv2d(3, width_list[0][0][0], kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(width_list[0][0][0])
            self.inplanes = width_list[0][0][0]
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
        self.relu_conv = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], width_list[0])
        self.layer2 = self._make_layer(block, 128, layers[1], width_list[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], width_list[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], width_list[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)

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
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        width_list: list,
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
            if width_list is None:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            elif width_list[0][0] is None:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, width_list[0][-1], stride),
                    norm_layer(width_list[0][-1]),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(width_list[0][0], width_list[0][-1], stride),
                    norm_layer(width_list[0][-1]),
                )

        layers = []
        if width_list is None:
            layers.append(
                block(
                    self.inplanes, planes, None, stride, downsample, self.groups, self.base_width, previous_dilation,
                    norm_layer
                )
            )
        else:
            layers.append(
                block(
                    self.inplanes, planes, width_list[0], stride, downsample, self.groups, self.base_width,
                    previous_dilation, norm_layer
                )
            )
        if downsample is not None and width_list is not None:
            self.inplanes = width_list[0][-1]
        else:
            self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if width_list is None:
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        None,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    )
                )
            else:
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        width_list[i],
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    )
                )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu_conv(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def ResNet18_flexi(plane_list):
    return ResNet_flexi(BasicBlock_flexi, [2, 2, 2, 2], plane_list)

def ResNet34_flexi(plane_list):
    return ResNet_flexi(BasicBlock_flexi, [3, 4, 6, 3], plane_list)

def ResNet50_flexi(plane_list):
    return ResNet_flexi(Bottleneck_flexi, [3, 4, 6, 3], plane_list)

def Wide_ResNet50_2_flexi(plane_list):
    return ResNet_flexi(Bottleneck_flexi, [3, 4, 6, 3], plane_list,  width_per_group=128)

def get_axis_to_perm_ResNet50():
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,), f"{name}.running_mean": (p,),
                            f"{name}.running_var": (p,)}
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
        #**conv('conv1', None, 'relu_conv'),
        #**norm('bn1', 'relu_conv'),

        #**bottleneck('layer1.0', 'relu_conv', 'layer1.0.relu3'),
        #**conv('layer1.0.downsample.0', 'relu_conv', 'layer1.0.relu3'),
        #**norm('layer1.0.downsample.1', 'layer1.0.relu3'),
        #**bottleneck('layer1.1', 'layer1.0.relu3', 'layer1.0.relu3'),
        #**bottleneck('layer1.2', 'layer1.0.relu3', 'layer1.0.relu3'),

        #**bottleneck('layer2.0', 'layer1.0.relu3', 'layer2.0.relu3'),
        #**conv('layer2.0.downsample.0', None, 'layer2.0.relu3'),
        #**norm('layer2.0.downsample.1', 'layer2.0.relu3'),
        #**bottleneck('layer2.1', 'layer2.0.relu3', 'layer2.0.relu3'),
        #**bottleneck('layer2.2', 'layer2.0.relu3', 'layer2.0.relu3'),
        #**bottleneck('layer2.3', 'layer2.0.relu3', 'layer2.0.relu3'),

        #**bottleneck('layer3.0', 'layer2.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.0', None, 'layer3.0.relu3'),
        **conv('layer3.0.downsample.0', None, 'layer3.0.relu3'),
        **norm('layer3.0.downsample.1', 'layer3.0.relu3'),
        **bottleneck('layer3.1', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.2', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.3', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.4', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.4', 'layer3.0.relu3', 'layer3.0.relu3'),
        **bottleneck('layer3.5', 'layer3.0.relu3', 'layer3.0.relu3'),

        **bottleneck('layer4.0', 'layer3.0.relu3', 'layer4.0.relu3'),
        **conv('layer4.0.downsample.0', 'layer3.0.relu3', 'layer4.0.relu3'),
        **norm('layer4.0.downsample.1', 'layer4.0.relu3'),
        **bottleneck('layer4.1', 'layer4.0.relu3', 'layer4.0.relu3'),
        **bottleneck('layer4.2', 'layer4.0.relu3', 'layer4.0.relu3'),

        **dense('fc', 'layer4.0.relu3', None)
    }

    return axis_to_perm

def get_axis_to_perm_ResNet18():
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}
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
        #**conv('conv1', None, 'relu_conv'),
        #**norm('bn1', 'relu_conv'),

        #**bottleneck('layer1.0', 'relu_conv', 'layer1.0.relu3'),
        #**conv('layer1.0.downsample.0', 'relu_conv', 'layer1.0.relu3'),
        #**norm('layer1.0.downsample.1', 'layer1.0.relu3'),
        #**bottleneck('layer1.1', 'layer1.0.relu3', 'layer1.0.relu3'),
        #**bottleneck('layer1.2', 'layer1.0.relu3', 'layer1.0.relu3'),

        #**bottleneck('layer2.0', 'layer1.0.relu3', 'layer2.0.relu3'),
        #**conv('layer2.0.downsample.0', None, 'layer2.0.relu3'),
        #**norm('layer2.0.downsample.1', 'layer2.0.relu3'),
        #**bottleneck('layer2.1', 'layer2.0.relu3', 'layer2.0.relu3'),
        #**bottleneck('layer2.2', 'layer2.0.relu3', 'layer2.0.relu3'),
        #**bottleneck('layer2.3', 'layer2.0.relu3', 'layer2.0.relu3'),

        #**bottleneck('layer3.0', 'layer2.0.relu3', 'layer3.0.relu3'),
        **basicblock('layer3.0', None, 'layer3.0.relu2'),
        **conv('layer3.0.downsample.0', None, 'layer3.0.relu2'),
        **norm('layer3.0.downsample.1', 'layer3.0.relu2'),
        **basicblock('layer3.1', 'layer3.0.relu2', 'layer3.0.relu2'),

        **basicblock('layer4.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **conv('layer4.0.downsample.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **norm('layer4.0.downsample.1', 'layer4.0.relu2'),
        **basicblock('layer4.1', 'layer4.0.relu2', 'layer4.0.relu2'),

        **dense('fc', 'layer4.0.relu2', None)
    }

    return axis_to_perm

def get_axis_to_perm_ResNet34():
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}
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
        #**conv('conv1', None, 'relu_conv'),
        #**norm('bn1', 'relu_conv'),

        #**bottleneck('layer1.0', 'relu_conv', 'layer1.0.relu3'),
        #**conv('layer1.0.downsample.0', 'relu_conv', 'layer1.0.relu3'),
        #**norm('layer1.0.downsample.1', 'layer1.0.relu3'),
        #**bottleneck('layer1.1', 'layer1.0.relu3', 'layer1.0.relu3'),
        #**bottleneck('layer1.2', 'layer1.0.relu3', 'layer1.0.relu3'),

        #**bottleneck('layer2.0', 'layer1.0.relu3', 'layer2.0.relu3'),
        #**conv('layer2.0.downsample.0', None, 'layer2.0.relu3'),
        #**norm('layer2.0.downsample.1', 'layer2.0.relu3'),
        #**bottleneck('layer2.1', 'layer2.0.relu3', 'layer2.0.relu3'),
        #**bottleneck('layer2.2', 'layer2.0.relu3', 'layer2.0.relu3'),
        #**bottleneck('layer2.3', 'layer2.0.relu3', 'layer2.0.relu3'),

        #**bottleneck('layer3.0', 'layer2.0.relu3', 'layer3.0.relu3'),
        **basicblock('layer3.0', None, 'layer3.0.relu2'),
        **conv('layer3.0.downsample.0', None, 'layer3.0.relu2'),
        **norm('layer3.0.downsample.1', 'layer3.0.relu2'),
        **basicblock('layer3.1', 'layer3.0.relu2', 'layer3.0.relu2'),
        **basicblock('layer3.2', 'layer3.0.relu2', 'layer3.0.relu2'),
        **basicblock('layer3.3', 'layer3.0.relu2', 'layer3.0.relu2'),
        **basicblock('layer3.4', 'layer3.0.relu2', 'layer3.0.relu2'),
        **basicblock('layer3.5', 'layer3.0.relu2', 'layer3.0.relu2'),

        **basicblock('layer4.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **conv('layer4.0.downsample.0', 'layer3.0.relu2', 'layer4.0.relu2'),
        **norm('layer4.0.downsample.1', 'layer4.0.relu2'),
        **basicblock('layer4.1', 'layer4.0.relu2', 'layer4.0.relu2'),
        **basicblock('layer4.2', 'layer4.0.relu2', 'layer4.0.relu2'),

        **dense('fc', 'layer4.0.relu2', None)
    }

    return axis_to_perm

def merge_channel_ResNet50(model_param, max_ratio=1., threshold=0.1):
    axes_to_perm = get_axis_to_perm_ResNet50()
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight(perm_to_axes, model_param, max_ratio, threshold)
    #p_in_list = ['relu_conv', 'layer1.0.relu3', 'layer2.0.relu3', 'layer3.0.relu3']
    p_in_list = ['relu_conv', 'layer1.0.relu3', 'layer2.0.relu3', 'layer3.0.relu3']
    plane_list = [None, None] + [[] for _ in range(2)]
    layers_n_block = [3, 4, 6, 3]
    for i in range(2, 4):
        layer = i + 1
        for j in range(layers_n_block[i]):
            block_plane_list = []
            if j == 0 and i >2:
                block_plane_list.append(perm_size[p_in_list[i]])
            else:
                block_plane_list.append(None)
            block_plane_list.append(perm_size[f'layer{layer}.{j}.relu1'])
            block_plane_list.append(perm_size[f'layer{layer}.{j}.relu2'])
            if j == 0:
                block_plane_list.append(perm_size[f'layer{layer}.0.relu3'])
            else:
                block_plane_list.append(None)
            plane_list[i].append(block_plane_list)
    model = ResNet50_flexi(plane_list)
    model.load_state_dict(param)
    print(plane_list)
    return model

def merge_channel_WRN(model_param, max_ratio=1., threshold=0.1):
    axes_to_perm = get_axis_to_perm_ResNet50()
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight(perm_to_axes, model_param, max_ratio, threshold)
    #p_in_list = ['relu_conv', 'layer1.0.relu3', 'layer2.0.relu3', 'layer3.0.relu3']
    p_in_list = ['relu_conv', 'layer1.0.relu3', 'layer2.0.relu3', 'layer3.0.relu3']
    plane_list = [None, None] + [[] for _ in range(2)]
    layers_n_block = [3, 4, 6, 3]
    for i in range(2, 4):
        layer = i + 1
        for j in range(layers_n_block[i]):
            block_plane_list = []
            if j == 0 and i >2:
                block_plane_list.append(perm_size[p_in_list[i]])
            else:
                block_plane_list.append(None)
            block_plane_list.append(perm_size[f'layer{layer}.{j}.relu1'])
            block_plane_list.append(perm_size[f'layer{layer}.{j}.relu2'])
            if j == 0:
                block_plane_list.append(perm_size[f'layer{layer}.0.relu3'])
            else:
                block_plane_list.append(None)
            plane_list[i].append(block_plane_list)
    model = Wide_ResNet50_2_flexi(plane_list)
    model.load_state_dict(param)
    print(plane_list)
    return model

def merge_channel_ResNet18(model_param, max_ratio=1., threshold=0.1):
    axes_to_perm = get_axis_to_perm_ResNet18()
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight(perm_to_axes, model_param, max_ratio, threshold)
    #p_in_list = ['relu_conv', 'layer1.0.relu3', 'layer2.0.relu3', 'layer3.0.relu3']
    p_in_list = ['relu_conv', 'layer1.0.relu2', 'layer2.0.relu2', 'layer3.0.relu2']
    plane_list = [None, None] + [[] for _ in range(2)]
    layers_n_block = [2, 2, 2, 2]
    for i in range(2, 4):
        layer = i + 1
        for j in range(layers_n_block[i]):
            block_plane_list = []
            if j == 0 and i > 2:
                block_plane_list.append(perm_size[p_in_list[i]])
            else:
                block_plane_list.append(None)
            block_plane_list.append(perm_size[f'layer{layer}.{j}.relu1'])
            if j == 0:
                block_plane_list.append(perm_size[f'layer{layer}.0.relu2'])
            else:
                block_plane_list.append(None)
            plane_list[i].append(block_plane_list)
    model = ResNet18_flexi(plane_list)
    model.load_state_dict(param)
    print(plane_list)
    return model

def merge_channel_ResNet34(model_param, max_ratio=1., threshold=0.1):
    axes_to_perm = get_axis_to_perm_ResNet34()
    perm_to_axes = axes2perm_to_perm2axes(axes_to_perm)
    param, perm_size = self_merge_weight(perm_to_axes, model_param, max_ratio, threshold)
    #p_in_list = ['relu_conv', 'layer1.0.relu3', 'layer2.0.relu3', 'layer3.0.relu3']
    p_in_list = ['relu_conv', 'layer1.0.relu2', 'layer2.0.relu2', 'layer3.0.relu2']
    plane_list = [None, None] + [[] for _ in range(2)]
    layers_n_block = [3, 4, 6, 3]
    for i in range(2, 4):
        layer = i + 1
        for j in range(layers_n_block[i]):
            block_plane_list = []
            if j == 0 and i > 2:
                block_plane_list.append(perm_size[p_in_list[i]])
            else:
                block_plane_list.append(None)
            block_plane_list.append(perm_size[f'layer{layer}.{j}.relu1'])
            if j == 0:
                block_plane_list.append(perm_size[f'layer{layer}.0.relu2'])
            else:
                block_plane_list.append(None)
            plane_list[i].append(block_plane_list)
    model = ResNet34_flexi(plane_list)
    model.load_state_dict(param)
    print(plane_list)
    return model

# Old weights with accuracy 76.130%
#resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# New weights with accuracy 80.858%
#resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
if __name__ =="__main__":
    model = ResNet50()
    '''for name, module in model.named_modules():
        print(name)'''
    for key in model.state_dict().keys():
        print(key)