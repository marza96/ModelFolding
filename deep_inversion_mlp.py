'''
ResNet model inversion for CIFAR10.

Copyright (C) 2020 NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License (1-Way Commercial). To view a copy of this license, visit https://github.com/NVlabs/DeepInversion/blob/master/LICENSE
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import os
import glob
import collections

#provide intermeiate information
debug_output = False
debug_output = True


class DeepInversionFeatureHook():
    '''
    Impleme    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0])
        var = input[0].contiguous().var(0, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def get_images(net, bs=32, epochs=1000, idx=-1, var_scale=0.00005,
               prefix=None, competitive_scale=0.01, train_writer = None, global_iteration=None,
               use_amp=False,
               optimizer = None, inputs = None, bn_reg_scale = 0.0, random_labels = True, l2_coeff=0.0, file_name="dummy.pt"):

    # preventing backpropagation through student for Adaptive DeepInversion
    best_cost = 1e26

    # initialize gaussian inputs
    # inputs.data = torch.randn((bs, 3, 224, 224), requires_grad=True, device='cuda')
    # if use_amp:
    #     inputs.data = inputs.data.half()

    # set up criteria for optimization
    criterion = nn.CrossEntropyLoss()

    optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer

    # target outputs to generate
    # if random_labels:
    # else:
    #     targets = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 25 + [0, 1, 2, 3, 4, 5]).to('cuda')
    targets = torch.LongTensor([random.randint(0,9) for _ in range(bs)]).to('cuda')


    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm1d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # setting up the range for jitter
    lim_0, lim_1 = 2, 2


    for epoch in range(epochs):
        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))

        # foward with jit images
        optimizer.zero_grad()
        net.zero_grad()
        outputs = net(inputs_jit)
        loss = criterion(outputs, targets)
        loss_target = loss.item()

        # apply total variation regularization
        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
        diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss = loss + var_scale * loss_var


        # R_feature loss
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
        loss = loss + bn_reg_scale * loss_distr # best for noise before BN

        # l2 loss
        if 1:
            loss = loss + l2_coeff * torch.norm(inputs_jit, 2)

        if debug_output and epoch % 200==0:
            print(f"It {epoch}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")
            vutils.save_image(inputs.data.clone(),
                              './{}/output_{}.png'.format(prefix, epoch//200),
                              normalize=True, scale_each=True, nrow=10)

        if best_cost > loss.item():
            best_cost = loss.item()
            best_inputs = inputs.data

        # backward pass
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

    outputs=net(best_inputs)
    _, predicted_teach = outputs.max(1)

    if idx == 0:
        print('Teacher correct out of {}: {}, loss at {}'.format(bs, predicted_teach.eq(targets).sum().item(), criterion(outputs, targets).item()))

    name_use = "best_images"
    if prefix is not None:
        name_use = prefix + name_use
    next_batch = len(glob.glob("./%s/*.png" % name_use)) // 1

    vutils.save_image(best_inputs[:20].clone(),
                      './{}/output_{}.png'.format(name_use, next_batch),
                      normalize=True, scale_each = True, nrow=10)

    torch.save(best_inputs, file_name)

    return best_inputs



class Args:
    def __init__(args):
        args.bs = 256
        args.iters_mi = 4000
        args.cig_scale = 0.0
        args.di_lr = 0.0015
        args.di_var_scale = 2.5e-6
        args.di_l2_scale = 0.2
        args.r_feature_weight = 1e4
        args.amp = False
        args.exp_descr = "try3"
        args.teacher_weights = "/home/m/marza1/Iterative-Feature-Merging/checkpoints/mlp_3Xwider_CIFAR10.pt"


from model.mlp import MLP
def main():
    args = Args()

    net_teacher = MLP(wider_factor=3)
    w = torch.load(args.teacher_weights, map_location="cpu")
    net_teacher.load_state_dict(w)
    net_teacher.to("cuda")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net_teacher = net_teacher.to(device)

    criterion = nn.CrossEntropyLoss()

    data_type = torch.half if args.amp else torch.float
    inputs = torch.randn((args.bs, 3, 32, 32), requires_grad=True, device='cuda', dtype=data_type)

    optimizer_di = optim.Adam([inputs], lr=args.di_lr)


    net_teacher.to("cuda").eval() #important, otherwise generated images will be non natural

    cudnn.benchmark = True


    batch_idx = 0
    prefix = "runs/data_generation/"+args.exp_descr+"/"

    for create_folder in [prefix, prefix+"/best_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

    train_writer = None  # tensorboard writter
    global_iteration = 0

    print("Starting model inversion")

    net_teacher.eval()
    inputs = get_images(net=net_teacher, bs=args.bs, epochs=args.iters_mi, idx=batch_idx,
                        prefix=prefix, competitive_scale=args.cig_scale,
                        train_writer=train_writer, global_iteration=global_iteration, use_amp=args.amp,
                        optimizer=optimizer_di, inputs=inputs, bn_reg_scale=args.r_feature_weight,
                        var_scale=args.di_var_scale, random_labels=False, l2_coeff=args.di_l2_scale, file_name="mlp3x_cifar10.pt")


if __name__ == "__main__":
    main()
