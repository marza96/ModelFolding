from collections import defaultdict
from re import L
from typing import NamedTuple
import time
import copy
import re

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from hkmeans import HKMeans
from scipy import spatial

import itertools
import matplotlib.pyplot as plt

def axes2perm_to_perm2axes(axes_to_perm):
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return perm_to_axes

# def self_weight_matching(perm_to_axes, params):
#   perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
#   distances = {}
#   for p_name in perm_sizes.keys():
#       n = perm_sizes[p_name]
#       A = torch.zeros((n, n)).cuda()
#       for wk, axis in perm_to_axes[p_name]:
#           w_a = params[wk]
#           #w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
#           w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
#           #w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

#           A += w_a @ w_a.T
#       distance_matrix = A + A.T - A.diag().unsqueeze(0) - A.diag().unsqueeze(1)
#       distances[p_name] = distance_matrix.cpu()
#   return distances

def self_weight_matching_p_name(perm_to_axes, params, p_name, n):
  A = torch.zeros((n, n)).cuda()
  for wk, axis in perm_to_axes[p_name]:
    w_a = params[wk]
    if len(w_a.shape) < 2 or "identity_transform" in wk:
        pass
    else:
        print(wk, axis)
        #w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
        w_a = torch.movedim(w_a, axis, 0).reshape((n, -1))
        # w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

        A += w_a @ w_a.T
  distance_matrix = A + A.T - A.diag().unsqueeze(0) - A.diag().unsqueeze(1)
  
  return distance_matrix.cpu()
  
def self_weight_matching_p_name_euclidean(perm_to_axes, params, p_name, n):
  A = torch.zeros((n, n)).cuda()
  for wk, axis in perm_to_axes[p_name]:
    w_a = params[wk]
    if len(w_a.shape) < 2 or "identity_transform" in wk:
        pass
    else:
        #w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
        w_a = torch.movedim(w_a, axis, 0).reshape((n, -1))
        # w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

        A += torch.cdist(w_a, w_a) ** 2 # w_a @ w_a.T
  distance_matrix = A

  return distance_matrix.cpu()

def self_weight_matching_p_name_raw_vec(perm_to_axes, params, merges, p_name, n):
  A = None
#   print(".........")
  for wk, axis in perm_to_axes[p_name]:
    w_a = params[wk]

    if len(w_a.shape) < 2 or "identity_transform" in wk : #or "bn" in wk
    # if "running" in wk or "identity_transform" in wk : #or "bn" in wk
        pass
    else:
        # print("wk", wk, "NOTE YOU HAVE CHANGED THE COST")
        w_a = torch.movedim(w_a, axis, 0).reshape((n, -1))

        if A is None:
            A = w_a
        else:
            A = torch.hstack((A, w_a))

  return A.cpu()

def merge_channel_p_name_clustering(perm_to_axes, params, merges, p_name, merge):
    params = copy.deepcopy(params)
    temp = None
    print(".........", p_name)
    for wk, axis in perm_to_axes[p_name]:
        if wk == "conv1.weightdd" or wk == "conv1.bndd" in wk:
            pass
        else:
            assert axis in (0, 1)
            if axis == 0:
                # print("AX 0", wk)
                p = params[wk].detach().clone()
                if len(p.shape) == 1:
                    params[wk] = torch.diag(1.0 / torch.diag(merge @ merge.T)) @ merge @ p 
                else:
                    p_np = params[wk].detach().cpu().numpy()

                    sh = p.shape
                    merged =  torch.diag(1.0 / torch.diag(merge @ merge.T)) @ merge @ p.clone().detach().reshape(sh[0], -1)
                    merged = merged.reshape(merge.shape[0], *sh[1:])
                    params[wk] = merged
            else:
                # print("AX 1", wk)
                p = params[wk].detach().clone()
  
                if len(p.shape) == 2:
                    p = p.permute(1, 0)
                else:
                    p = p.permute(1, 0, 2, 3)

                sh = p.shape
                merged =   merge @ p.clone().detach().reshape(sh[0], -1)
                
                merged = merged.reshape(merge.shape[0], *sh[1:])

                if len(p.shape) == 2:
                    merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0)
                else:
                    merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0, 2, 3)

                params[wk] = merged

    return params

def merge_channel_p_name(perm_to_axes, params, merge_num, p_name, n, i, j):
    min_ij = min(i, j)
    max_ij = max(i, j)
    indices = [num for num in range(min_ij)] + [num for num in range(min_ij + 1, max_ij)] + [num for num in range(max_ij + 1, n)] + [max_ij, min_ij]
    assert len(indices) == n
    merge_num_list = merge_num[p_name]
    merge_num_list = merge_num_list[indices]
    for wk, axis in perm_to_axes[p_name]:
        if wk == "conv1.weightdd" or wk == "conv1.bnddd" in wk:
            pass
        else:
            w_a = params[wk]
            assert axis in (0, 1)
            if axis == 0:
                w_a = w_a[indices]
                w_a[-2] += w_a[-1]
                params[wk] = w_a[:-1]
            else:
                w_a = w_a[:, indices]
                w_a[:, -2] = (w_a[:, -2] * merge_num_list[-2] + w_a[:, -1] * merge_num_list[-1]) / (
                            merge_num_list[-2] + merge_num_list[-1])
                params[wk] = w_a[:, :-1]

    merge_num_list[-2] += merge_num_list[-1]
    merge_num[p_name] =  merge_num_list[:-1]

    return params, merge_num


class KMeansWM:
    def __init__(self, n_clusters, n_features, normalize=False):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.normalize  = normalize

    def __call__(self, weight):
        # km = SpectralClustering(
        #     n_clusters=self.n_clusters,
        #     affinity="nearest_neighbors",
        #     n_neighbors=10,
        #     assign_labels="cluster_qr",
        #     eigen_solver=None,
        # )

        km = AgglomerativeClustering(n_clusters=self.n_clusters)

        # km = HKMeans(n_clusters=self.n_clusters, random_state=None, n_init=1,
        #           n_jobs=-1, max_iter=5, verbose=True)

        km.fit(weight.cpu().numpy())

        matches = torch.zeros((self.n_features, self.n_clusters), device="cuda")

        for model_idx, match_idx in enumerate(km.labels_):
            matches[model_idx, match_idx] = 1
                                
        merge = matches.detach().clone() 
        unmerge = matches.detach().clone()  

        return merge.T, unmerge


def self_merge_weight_clustering(perm_to_axes, params, max_ratio=0.5, threshold=0.1, hooks=None):
    perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    merge_num = {p: torch.ones(perm_sizes[p]) for p in perm_sizes.keys()}

    for p_name in perm_sizes.keys():
        print('"' + p_name + '"')
        ratio = max_ratio
        n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]

        if hooks is not None:
            distance = hooks[p_name].get_features().permute(1, 0, 2, 3)
            distance = distance.reshape(distance.shape[0], -1)
        else:
            distance = self_weight_matching_p_name_raw_vec(perm_to_axes, params, p_name, n)

        merger = KMeansWM(int(distance.shape[0] * ratio), distance.shape[0], normalize=False)
        merge, _ = merger(distance)
        prm = merge_channel_p_name_clustering(perm_to_axes, params, p_name, merge)

        for wk, axis in perm_to_axes[p_name]:
            params[wk] = prm[wk]

    new_perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    return params, new_perm_sizes


def self_merge_weight_dummy(perm_to_axes, params, max_ratio=0.5, threshold=0.1):
        # print(p_name)

        # # ratio = 1.0
        # # if "layer4" in p_name:
        # #     ratio = 0.5

        # ratio = 0.5

        # n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]
        # distance = self_weight_matching_p_name(perm_to_axes, params, p_name, n)

        # merger = KMeansWM(int(distance.shape[0] * ratio), distance.shape[0], normalize=False)
        # merge, _ = merger(distance)

        # if ("layer3.1.conv1.weight", 0) in perm_to_axes[p_name]:
        #     params["layer3.1.conv1.weight"] = torch.eye(params["layer3.1.conv1.weight"].shape[0]).to("cuda")
        #     sh = params["layer3.1.conv1.weight"].shape
        #     merged_dbg = merge @ params["layer3.1.conv1.weight"].clone().detach().reshape(sh[0], -1)
        #     merged_dbg = merged_dbg.reshape(merge.shape[0], *sh[1:])
        #     print("DBG SHAPE FIRST", merged_dbg.shape, params["layer3.1.conv1.weight"].shape)

        # orig_merge = merge.detach().clone()

        # dest_indices = torch.argmax(merge, dim=1)

        # for idx, dest in enumerate(dest_indices):
        #     merge[idx, dest] = 0

        # for idx in range(len(dest_indices)):
        #     for i in range(int(merge[idx, :].sum())):
        #         dest = dest_indices[idx]
        #         n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]

        #         source = torch.argmax(merge[idx, :]) 
        #         params, merge_num = merge_channel_p_name(perm_to_axes, params, merge_num, p_name, n, dest, source)

        #         merge = torch.hstack((merge[:, :source], merge[:, source + 1:]))
        #         orig_merge = torch.hstack((orig_merge[:, :source], orig_merge[:, source + 1:]))
                
        #         dest_indices = torch.argmax(orig_merge, dim=1)

        # if ("layer3.1.conv1.weight", 0) in perm_to_axes[p_name]:
        #     zz = params["layer3.1.conv1.weight"][:, :]

        #     print("DBG SHAPE SECOND", merged_dbg.shape, params["layer3.1.conv1.weight"].shape)
        #     print("DBG VALUE", torch.norm(merged_dbg - params["layer3.1.conv1.weight"]))
        #     plt.figure()

        #     plt.imshow((merged_dbg[:, :].T @ merged_dbg[:, :]).detach().cpu().numpy())
        #     # rp = torch.randperm(zz.shape[0])
        #     # plt.imshow((zz[rp, :].T @ zz[rp, :]).detach().cpu().numpy())
        #     plt.savefig("A")

        #     plt.figure()
        #     plt.imshow((zz.T @ zz).detach().cpu().numpy())
        #     plt.savefig("B")
        #     plt.savefib("K")

    new_perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    return params, new_perm_sizes
    

def self_merge_weight(perm_to_axes, params, max_ratio=0.5, threshold=0.1):
    perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    merge_num = {p: torch.ones(perm_sizes[p]) for p in perm_sizes.keys()}
    iter = 0
    time_used = []
    tick = time.time()
    while True:
        Flag = True
        for p_name in perm_sizes.keys():
            n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]
            if iter >= perm_sizes[p_name] * max_ratio:
                pass
            else:
                distance = self_weight_matching_p_name(perm_to_axes, params, p_name, n)
                value, indices = torch.topk(distance.view(-1), k=n + 1)
                v = value[-1]
                if v.item() < threshold * torch.min(distance).item():
                    pass
                else:
                    Flag = False
                    indices = indices[-1].item()
                    assert v <= 0
                    i = indices // n
                    j = int(indices % n)
                    assert distance[i, j].item() == v.item()
                    params, merge_num = merge_channel_p_name(perm_to_axes, params, merge_num, p_name, n, i, j)

        iter += 1
        '''tock = time.time()
        time_used.append(tock - tick)
        tick = tock'''
        #print(f"iter: {iter}, time_used: {time.time() - tick:.2f}s")
        if Flag:
            break
        '''if len(time_used) >=100:
            break
    print(f"mean:{np.mean(time_used)}; std:{np.std(time_used)}")'''
    new_perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    return params, new_perm_sizes

def self_merge_weight_get_dict(perm_to_axes, params, max_ratio=0.5, threshold=0.1):
    perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    merge_num = {p: torch.ones(perm_sizes[p]) for p in perm_sizes.keys()}
    distinct_features = {p: [[i] for i in range(perm_sizes[p])] for p in perm_sizes.keys()}
    iter = 0
    tick = time.time()
    while True:
        Flag = True
        for p_name in perm_sizes.keys():
            n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]
            if iter >= perm_sizes[p_name] * max_ratio:
                pass
            else:
                distance = self_weight_matching_p_name(perm_to_axes, params, p_name, n)
                value, indices = torch.topk(distance.view(-1), k=n + 1)
                v = value[-1]
                if v.item() < threshold * torch.min(distance).item():
                    pass
                else:
                    Flag = False
                    indices = indices[-1].item()
                    assert v <= 0
                    i = indices // n
                    j = int(indices % n)
                    assert distance[i, j].item() == v.item()
                    params, merge_num = merge_channel_p_name(perm_to_axes, params, merge_num, p_name, n, i, j)
                    min_ij = min(i, j)
                    max_ij = max(i, j)
                    list_i = distinct_features[p_name].pop(min_ij)
                    list_j = distinct_features[p_name].pop(max_ij-1)
                    distinct_features[p_name].append(list_i+list_j)
        iter += 1
        print(f"iter: {iter}, time_used: {time.time() - tick:.2f}s")
        if Flag:
            break
    new_perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    return params, new_perm_sizes, distinct_features