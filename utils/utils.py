import numpy as np
import time
import sys
import torch
import torch.nn as nn
import random
from torchvision import transforms
import copy
from collections import defaultdict
from matplotlib import cm
from PIL import Image
import random


class ConvBnormFuse(torch.nn.Module):
    def __init__(self, conv, bnorm):
        super().__init__()
        self.fused = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )
        self.weight = self.fused.weight
        self.bias = self.fused.bias

        self._fuse(conv, bnorm)

    def _fuse(self, conv, bn):
        w_conv = conv.weight.clone().reshape(conv.out_channels, -1).detach() #view umjesto reshape
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var))).detach()

        w_bn.requires_grad = False
        w_conv.requires_grad = False
        
        ww = torch.mm(w_bn.detach(), w_conv.detach())
        ww.requires_grad = False
        self.fused.weight.data = ww.data.view(self.fused.weight.detach().size()).detach() 
 
        if conv.bias is not None:
            b_conv = conv.bias.detach()
        else:
            b_conv = torch.zeros( conv.weight.size(0), device=conv.weight.device )

        bn.bias.requires_grad = False
        bn.weight.requires_grad = False

        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        
        bb = ( torch.matmul(w_bn, b_conv) + b_bn ).detach()
        self.fused.bias.data = bb.data

    def forward(self, x):
        return self.fused(x)


def flatten_params(model):
  return model.state_dict()

def lerp(lam, t1, t2):
  t3 = copy.deepcopy(t2)
  for p in t1:
    t3[p] = (1 - lam) * t1[p] + lam * t2[p]
  return t3

def perm_to_axes_from_axes_to_perm(axes_to_perm: dict):
  perm_to_axes = defaultdict(list)
  for wk, axis_perms in axes_to_perm.items():
    for axis, perm in enumerate(axis_perms):
      if perm is not None:
        perm_to_axes[perm].append((wk, axis))
  return dict(perm_to_axes)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_train_resize = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(384),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

transform_test_resize = transforms.Compose([
    transforms.Resize(384),
    transforms.ToTensor(),
])

transform_train_norm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

transform_test_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

ImageNet_train_transform = transforms.Compose ( [
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
] )

ImageNet_test_transform = transforms.Compose ( [
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
] )

TinyImageNet_train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)
TinyImageNet_test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor()
    ]
)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def equalize_weight_norm(tensor, iteration=100):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    with torch.no_grad():
        for _ in range(iteration):
            if dimensions == 2:
                input_first_half = tensor[:num_output_fmaps // 2, :]
                input_second_half = tensor[num_output_fmaps // 2:, :]
                norm_input_first_half = torch.norm(input_first_half, dim=0)
                norm_input_second_half = torch.norm(input_second_half, dim=0)
                input_first_half_index = torch.argsort(norm_input_first_half, dim=0, descending=True)
                input_second_half_index = torch.argsort(norm_input_second_half, dim=0, descending=False)
                input_first_half = input_first_half[:, input_first_half_index]
                input_second_half = input_second_half[:, input_second_half_index]
                tensor = torch.cat([input_first_half, input_second_half], dim=0)

                output_first_half = tensor[:, :num_input_fmaps // 2]
                output_second_half = tensor[:, num_input_fmaps // 2:]
                norm_output_first_half = torch.norm(output_first_half, dim=1)
                norm_output_second_half = torch.norm(output_second_half, dim=1)
                output_first_half_index = torch.argsort(norm_output_first_half, dim=0, descending=True)
                output_second_half_index = torch.argsort(norm_output_second_half, dim=0, descending=False)
                output_first_half = output_first_half[output_first_half_index, :]
                output_second_half = output_second_half[output_second_half_index, :]
                tensor = torch.cat([output_first_half, output_second_half], dim=1)
            else:
                input_first_half = tensor[:num_output_fmaps // 2, :]
                input_second_half = tensor[num_output_fmaps // 2:, :]
                norm_input_first_half = torch.norm(
                    torch.moveaxis(input_first_half, 1, 0).reshape((num_input_fmaps, -1)), dim=1)
                norm_input_second_half = torch.norm(
                    torch.moveaxis(input_second_half, 1, 0).reshape((num_input_fmaps, -1)), dim=1)
                input_first_half_index = torch.argsort(norm_input_first_half, dim=0, descending=True)
                input_second_half_index = torch.argsort(norm_input_second_half, dim=0, descending=False)
                input_first_half = input_first_half[:, input_first_half_index]
                input_second_half = input_second_half[:, input_second_half_index]
                tensor = torch.cat([input_first_half, input_second_half], dim=0)

                output_first_half = tensor[:, :num_input_fmaps // 2]
                output_second_half = tensor[:, num_input_fmaps // 2:]
                norm_output_first_half = torch.norm(output_first_half.reshape((num_output_fmaps, -1)), dim=1)
                norm_output_second_half = torch.norm(output_second_half.reshape((num_output_fmaps, -1)), dim=1)
                output_first_half_index = torch.argsort(norm_output_first_half, dim=0, descending=True)
                output_second_half_index = torch.argsort(norm_output_second_half, dim=0, descending=False)
                output_first_half = output_first_half[output_first_half_index, :]
                output_second_half = output_second_half[output_second_half_index, :]
                tensor = torch.cat([output_first_half, output_second_half], dim=1)
    return tensor


def get_equalnorm_init(iteration):
    def equalnorm_init(m):
        if isinstance(m, nn.Conv2d):
            with torch.no_grad():
                m.weight.copy_(equalize_weight_norm(m.weight, iteration=iteration))
        elif isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(equalize_weight_norm(m.weight, iteration=iteration))
    return equalnorm_init

def magic_square_solution(tensor, iteration=2000, initial_temp=100.0, min_temp=0.001, cooling_rate=0.99):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    def score(tensor):
        if dimensions == 2:
            norm_input = torch.norm(tensor, dim=0)
            norm_output = torch.norm(tensor, dim=1)
        elif dimensions > 2:
            norm_input = torch.norm(torch.moveaxis(tensor, 1, 0).reshape((num_input_fmaps, -1)), dim=1)
            norm_output = torch.norm(tensor.reshape((num_output_fmaps, -1)), dim=1)
        else:
            raise ValueError("Tensor with fewer than 2 dimensions")
        return torch.var(norm_input, dim=0).item() + torch.var(norm_output, dim=0).item()

    current_solution = tensor.data.clone()
    current_score = score(current_solution)
    best_solution = tensor.data.clone()
    best_score = score(best_solution)
    current_temp = initial_temp
    for _ in range(iteration):
        x1, y1 = np.random.randint(0, num_output_fmaps), np.random.randint(0, num_input_fmaps)
        x2, y2 = np.random.randint(0, num_output_fmaps), np.random.randint(0, num_input_fmaps)
        tmp = current_solution[x1, y1]
        current_solution[x1, y1] = current_solution[x2, y2]
        current_solution[x2, y2] = tmp

        new_score = score(current_solution)

        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / current_temp):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_solution = current_solution.clone()
        else:
            tmp = current_solution[x1, y1]
            current_solution[x1, y1] = current_solution[x2, y2]
            current_solution[x2, y2] = tmp


        if current_temp <= min_temp:
            current_temp = min_temp
        else:
            current_temp *= cooling_rate


    return best_solution

def get_magic_square_init(iter=1000, initial_temp=100.0, min_temp=0.001, cooling_rate=0.99):
    def magic_square_init(m):
        if isinstance(m, nn.Conv2d):
            with torch.no_grad():
                m.weight.copy_(magic_square_solution(m.weight, iteration=iter, initial_temp=initial_temp, min_temp=min_temp, cooling_rate=cooling_rate))
        elif isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(magic_square_solution(m.weight, iteration=iter, initial_temp=initial_temp, min_temp=min_temp, cooling_rate=cooling_rate))
    return magic_square_init

def norm_sort_solution(param, perm2axis, same_direction):
    with torch.no_grad():
        for p, axes in perm2axis.items():
            for wk, axis in axes:
                w_a = param[wk].data.clone()
                n = w_a.shape[axis]
                if '.'.join(wk.split('.')[:-1] + ["running_mean"]) in param.keys():
                    pass
                else:
                    w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                    norm_weight = torch.norm(w_a, dim=1)
                    if axis == 0:
                        index = torch.argsort(norm_weight, dim=0, descending=True)
                        param[wk] = param[wk][index]
                    elif axis == 1:
                        index = torch.argsort(norm_weight, dim=0, descending=same_direction)
                        param[wk] = param[wk][:, index]
    return param

def norm_shuffle_solution(param, perm2axis):
    with torch.no_grad():
        for p, axes in perm2axis.items():
            for wk, axis in axes:
                w_a = param[wk].data.clone()
                n = w_a.shape[axis]
                if '.'.join(wk.split('.')[:-1] + ["running_mean"]) in param.keys():
                    pass
                else:
                    w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                    norm_weight = torch.norm(w_a, dim=1)
                    if axis == 0:
                        index = [i for i in range(n)]
                        random.shuffle(index)
                        param[wk] = param[wk][index]
                    elif axis == 1:
                        index = [i for i in range(n)]
                        random.shuffle(index)
                        param[wk] = param[wk][:, index]
    return param



def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)

import sys

class DupStdoutFileWriter(object):
    def __init__(self, stdout, path, mode):
        self.path = path
        self._content = ''
        self._stdout = stdout
        self._file = open(path, mode)

    def write(self, msg):
        while '\n' in msg:
            pos = msg.find('\n')
            self._content += msg[:pos + 1]
            self.flush()
            msg = msg[pos + 1:]
        self._content += msg
        if len(self._content) > 1000:
            self.flush()

    def flush(self):
        self._stdout.write(self._content)
        self._stdout.flush()
        self._file.write(self._content)
        self._file.flush()
        self._content = ''

    def __del__(self):
        self._file.close()

class DupStdoutFileManager(object):
    def __init__(self, path, mode='w+'):
        self.path = path
        self.mode = mode

    def __enter__(self):
        self._stdout = sys.stdout
        self._file = DupStdoutFileWriter(self._stdout, self.path, self.mode)
        sys.stdout = self._file

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout

from easydict import EasyDict as edict

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(indent_cnt=0)
def print_easydict(inp_dict: edict):
    for key, value in inp_dict.items():
        if type(value) is edict or type(value) is dict:
            print('{}{}:'.format(' ' * 2 * print_easydict.indent_cnt, key))
            print_easydict.indent_cnt += 1
            print_easydict(value)
            print_easydict.indent_cnt -= 1

        else:
            print('{}{}: {}'.format(' ' * 2 * print_easydict.indent_cnt, key, value))

@static_vars(indent_cnt=0)
def print_easydict_str(inp_dict: edict):
    ret_str = ''
    for key, value in inp_dict.items():
        if type(value) is edict or type(value) is dict:
            ret_str += '{}{}:\n'.format(' ' * 2 * print_easydict_str.indent_cnt, key)
            print_easydict_str.indent_cnt += 1
            ret_str += print_easydict_str(value)
            print_easydict_str.indent_cnt -= 1

        else:
            ret_str += '{}{}: {}\n'.format(' ' * 2 * print_easydict_str.indent_cnt, key, value)

    return ret_str

from matplotlib.colors import Normalize

def Cam_overlay(img, mask, max, colormap='jet', alpha=0.5,):
    cmap = cm.get_cmap(colormap)
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    #norm = Normalize(vmin=0, vmax=max)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

def transform_fft(img):
    if img.dim() > 3:
        img = img.view(-1, img.size(-3), img.size(-2), img.size(-1))
        result = []
        for i in range(img.size(0)):
            tmp = np.array(img[i].cpu())
            tmp = np.fft.fft2(tmp)
            tmp = np.fft.fftshift(tmp)
            result.append(torch.tensor(tmp).unsqueeze(0))
    else:
        img = np.array(img.cpu())
        img = np.fft.fft2(img)
        img = np.fft.fftshift(img)
        return torch.tensor(img)

    result = torch.cat(result, dim=0)
    return result

def transform_ifft(img):
    if img.dim() > 3:
        img = img.view(-1, img.size(-3), img.size(-2), img.size(-1))
        result = []
        for i in range(img.size(0)):
            tmp = np.array(img[i].cpu())
            tmp = np.fft.ifft2(np.fft.ifftshift(tmp))
            '''if np.abs(np.sum(np.imag(tmp))) > 1e-5:
                raise ValueError(f"imag of reconstructed image is too big:{np.abs(np.sum(np.imag(img)))}")'''
            result.append(torch.tensor(np.real(tmp)).float().unsqueeze(0))
    else:
        img = np.array(img.cpu())
        img = np.fft.ifft2(np.fft.ifftshift(img))
        '''if np.abs(np.sum(np.imag(img))) > 1e-5:
            raise ValueError(f"imag of reconstructed image is too big:{np.abs(np.sum(np.imag(img)))}")'''
        return torch.tensor(np.real(img)).float()

    result = torch.cat(result, dim=0)
    return result
