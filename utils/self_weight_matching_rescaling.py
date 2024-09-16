from collections import defaultdict
from re import L
from typing import NamedTuple
import time
import copy
import re
import torch.linalg
import torch.nn as nn

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn_extra.robust import RobustWeightedKMeans
from sklearn_extra.cluster import KMedoids, CLARA
from hkmeans import HKMeans
from scipy import spatial
from sklearn import preprocessing
import random
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

import itertools
import matplotlib.pyplot as plt

def axes2perm_to_perm2axes(axes_to_perm):
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return perm_to_axes


def self_weight_matching_p_name(perm_to_axes, params, p_name, n):
  A = torch.zeros((n, n)).cuda()
  for wk, axis in perm_to_axes[p_name]:
    w_a = params[wk]
    if len(w_a.shape) < 2 or "identity_transform" in wk:
        pass
    else:
        print(wk, axis, params[wk].shape)
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
        w_a = torch.movedim(w_a, axis, 0).reshape((n, -1))

        A += torch.cdist(w_a, w_a) ** 2
  distance_matrix = A

  return distance_matrix.cpu()


def self_weight_matching_p_name_raw_vec(perm_to_axes, params, p_name, n):
  A = None
  for wk, axis in perm_to_axes[p_name]:
    w_a = params[wk]

    if "running" in wk or "identity_transform" in wk:
        continue

    w_a = torch.movedim(w_a, axis, 0).reshape((n, -1))

    if A is None:
        A = w_a
    else:
        A = torch.hstack((A, w_a))

  return A.cpu()


def merge_channel_p_name_clustering(perm_to_axes, params, p_name, merge, scale):
    params = copy.deepcopy(params)
    temp = None
    for wk, axis in perm_to_axes[p_name]:
        if wk == "conv1.weightdd" or wk == "conv1.bndd" in wk:
            pass
        else:
            assert axis in (0, 1)
            if axis == 0:
                p = params[wk].detach().clone()
                if len(p.shape) == 1:
                    if "running_mean" in wk:
                        continue

                    if "running_var" in wk:
                        inv_stds = 1.0 / torch.sqrt(params[wk])  
                        new_inv_stds = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ inv_stds
                        new_var = torch.square(1.0 / (new_inv_stds))

                        key = ".".join(wk.split(".")[:-1]) + ".running_mean"
                        new_mean = params[key] * inv_stds
                        new_mean = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ new_mean
                        new_mean = new_mean * torch.sqrt(new_var)
    
                        params[wk] = new_var    
                        params[key] = new_mean

                        # params[wk] = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ (p) 
                    else:
                        params[wk] = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p 
                else:
                    sh = p.shape
                    merged = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p.clone().detach().reshape(sh[0], -1)
                    merged = merged.reshape(merge.shape[0], *sh[1:])
                    params[wk] = merged
            else:
                p = params[wk].detach().clone()

                print("ZZ", wk, p.shape)
                if "classifier.0" in wk:
                    print("UUUU", p.shape)
                    p = p.reshape(4096, 512, 7*7)
                    sh = p.shape
                    p = p.permute(1, 0, 2)
                    #p (512, 4096, 7*7)
                    p = p.reshape(sh[1], -1)
                    #p (512, 4096 * 7 * 7)

                    merged = merge @ scale @ p.clone().detach()

                    print("m1", merged.shape)
                    #p (r * 512, 4096 * 7 * 7)
                    merged = merged.reshape(merge.shape[0], 4096, 7 * 7)
                    print("m2", merged.shape)
                    #p (r * 512, 4096,  7 * 7)
                    merged = merged.permute(1, 0, 2)
                    merged = merged.reshape(4096, -1)
                    print("MMM", merged.shape)
                    params[wk] = merged
                else:
                    if len(p.shape) == 2:
                        p = p.permute(1, 0)
                    else:
                        p = p.permute(1, 0, 2, 3)
                
                    sh = p.shape
                    merged =  merge @ scale @ p.clone().detach().reshape(sh[0], -1)
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
    def __init__(self, n_clusters, n_features, normalize=False, use_kmeans=False):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.normalize  = normalize
        self.use_kmeans = use_kmeans
        
    def __call__(self, weight):
        km = AgglomerativeClustering(n_clusters=self.n_clusters)

        if self.use_kmeans is True:
            km = HKMeans(n_clusters=self.n_clusters, random_state=None, n_init=200,
                        n_jobs=-1, max_iter=10, verbose=True)
        
        X_scaled = weight.cpu().numpy()
        X = torch.tensor(X_scaled).to("cuda")

        if self.normalize is True:
            scaler = preprocessing.RobustScaler().fit(X_scaled)
            X_scaled = scaler.transform(weight.cpu().numpy())

        pca = PCA(n_components=weight.shape[0])
        X_scaled = pca.fit_transform(X_scaled)
        km.fit(X_scaled)

        matches = torch.zeros((self.n_features, self.n_clusters), device="cuda")

        for model_idx, match_idx in enumerate(km.labels_):
            matches[model_idx, match_idx] = 1
                                
        merge = matches.detach().clone() 

        m = merge.cpu().numpy()
        
        X_scaled = X_scaled / np.linalg.norm(X_scaled, axis=1)[:, None]
        inertia = (np.linalg.norm(
            X_scaled - m @ (np.diag(1.0 / np.diag(m.T @ m))) @ m.T @ X_scaled) ** 2) 

        M = np.sqrt(np.diag(1.0 / np.diag(m.T @ m))) @ m.T
        M = torch.tensor(M).to("cuda")
        X_scaled = torch.tensor(X_scaled).to("cuda")
        inner = X @ X.T
        scale =  torch.diag(torch.diag(inner @ M.T @ M)) @ torch.pinverse(torch.diag(torch.diag(M.T @ M @ inner @ M.T @ M)))

        scale = (torch.norm(X, dim=1, keepdim=True)) / torch.norm(M.T @ M @ X, dim=1, keepdim=True)
        scale =  torch.diag(scale[:, 0])
        print(scale)

        print(scale)
        return merge.T, scale


def self_merge_weight_clustering(perm_to_axes, params, max_ratio=0.5, threshold=0.1, hooks=None):
    perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}

    merges = dict()
    scales = dict()
    ratios = dict()

    for p_name in perm_sizes.keys():
        print('merging block: "' + p_name + '"')
        n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]

        if hooks is not None:
            distance = hooks[p_name].get_features().permute(1, 0, 2, 3)
            distance = distance.reshape(distance.shape[0], -1).detach().cpu()
        else:
            distance = self_weight_matching_p_name_raw_vec(perm_to_axes, params, p_name, n)
        
        merger = KMeansWM(int(distance.shape[0] * max_ratio), distance.shape[0], normalize=True)
        merge, scale = merger(distance)
        merges[p_name] = merge
        scales[p_name] = scale

    for p_name in perm_sizes.keys():
        merge = merges[p_name]
        scale = scales[p_name]
        prm = merge_channel_p_name_clustering(perm_to_axes, params, p_name, merge, scale)

        for wk, axis in perm_to_axes[p_name]:
            params[wk] = prm[wk]

    new_perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    return params, new_perm_sizes
    
    max_val = 0
    min_val = 1e10
    for p_name in perm_sizes.keys():
        if inertias[p_name] > max_val:
            max_val = inertias[p_name]
        if inertias[p_name] < min_val:
            min_val = inertias[p_name]

    for p_name in perm_sizes.keys():
        ratios[p_name] = max(max_ratio * 0.1, max_ratio * (1 - (inertias[p_name]) / ( max_val )))
        print("r", ratios[p_name], max_ratio * (1 - (inertias[p_name]) / ( max_val )), (1 - (inertias[p_name]) / ( max_val )))
        # print("i", inertias[p_name])

    for p_name in perm_sizes.keys():
        print('merging block: "' + p_name + '"')
        ratio = max_ratio
        n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]

        if hooks is not None:
            distance = hooks[p_name].get_features().permute(1, 0, 2, 3)
            distance = distance.reshape(distance.shape[0], -1).detach().cpu()
        else:
            distance = self_weight_matching_p_name_raw_vec(perm_to_axes, params, p_name, n)
        
        merger = KMeansWM(int(distance.shape[0] * ratios[p_name]), distance.shape[0], normalize=False)
        merge, inertia = merger(distance)
        merges[p_name] = merge
        inertias[p_name] = inertia

    for p_name in perm_sizes.keys():
        merge = merges[p_name]
        prm = merge_channel_p_name_clustering(perm_to_axes, params, p_name, merge)

        for wk, axis in perm_to_axes[p_name]:
            params[wk] = prm[wk]
    

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
        if Flag:
            break

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
