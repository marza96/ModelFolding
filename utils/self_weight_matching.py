from collections import defaultdict
import copy
import torch.nn as nn

import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, BisectingKMeans
from sklearn_extra.robust import RobustWeightedKMeans
from sklearn_extra.cluster import KMedoids, CLARA
from hkmeans import HKMeans
from sklearn import preprocessing
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


def self_weight_matching_p_name_raw_vec_d(perm_to_axes, params, p_name, n):
  A = None
  for wk, axis in perm_to_axes[p_name]:
    w_a = params[wk]

    if "running" in wk or "identity_transform" in wk:
        continue
    
    if axis == 1:
        continue

    if len(w_a.shape) < 2:
        continue

    w_a = torch.movedim(w_a, axis, 0).reshape((n, -1))

    if A is None:
        A = w_a
    else:
        A = torch.hstack((A, w_a))

  return A.cpu()



def merge_channel_p_name_clustering(perm_to_axes, params, p_name, merge):
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
                    print("DBG", wk)
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

                if "classifier.0" in wk:
                    print("UUUU", p.shape)
                    p = p.reshape(4096, 512, 7*7)
                    sh = p.shape
                    p = p.permute(1, 0, 2)
                    #p (512, 4096, 7*7)
                    p = p.reshape(sh[1], -1)
                    #p (512, 4096 * 7 * 7)

                    merged = merge @ p.clone().detach()

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
                    merged =  merge @ p.clone().detach().reshape(sh[0], -1)
                    merged = merged.reshape(merge.shape[0], *sh[1:])

                    if len(p.shape) == 2:
                        merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0)
                    else:
                        merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0, 2, 3)

                    params[wk] = merged

    return params


def get_average_correlation(merge, w):
    avg_corr = np.zeros(merge.shape[0])
    for i in range(avg_corr.shape[0]):
        wh = np.where(merge.cpu().numpy()[i, :] > 0)[0]
        pairs = list(itertools.product(wh, wh))
        
        assert len(wh) >= 1

        if len(wh) <= 1:
            continue

        pairs = [pair for pair in pairs if pair[0] != pair[1]]
        cnt = 0
        for pair in pairs:
            cnt += 1
            m = pair[0]
            n = pair[1]
            a = w[m, :].flatten()
            b = w[n, :].flatten()
            avg_corr[i] += (a.T @ b) / np.sqrt((a.T @ a) * (b.T @ b))
        
        avg_corr[i] /= cnt

    avg_corr = torch.tensor(avg_corr).to("cuda").float()

    return avg_corr


def merge_channel_p_name_clustering_approx_repair(perm_to_axes, params, p_name, merge):
    params = copy.deepcopy(params)
    n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]
    w = self_weight_matching_p_name_raw_vec_d(perm_to_axes, params, p_name, n)

    avg_corr = get_average_correlation(merge, w.detach().cpu().numpy())

    for wk, axis in perm_to_axes[p_name]:   
        if axis == 0:
            p = params[wk].detach().clone()
            if len(p.shape) == 1:
                if "running_mean" in wk:
                    params[wk] = torch.zeros(merge.shape[0])
                    continue

                if "running_var" in wk:
                    params[wk] = torch.ones(merge.shape[0])
                    continue

                params[wk] = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p

                if "weight" in wk:
                    n_c = torch.sum(merge, axis=1)
                    params[wk] = params[wk] * torch.sqrt(n_c / (1 + (n_c - 1) * avg_corr))
            else:
                sh = p.shape
                merged = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p.clone().detach().reshape(sh[0], -1)
                merged = merged.reshape(merge.shape[0], *sh[1:])
                params[wk] = merged

                if wk == "layer4.1.conv1.weight":
                    print(wk, merged.shape)
        else:
            p = params[wk].detach().clone()

            if "classifier.0" in wk:
                print("UUUU", p.shape)
                p = p.reshape(4096, 512, 7*7)
                sh = p.shape
                p = p.permute(1, 0, 2)
                #p (512, 4096, 7*7)
                p = p.reshape(sh[1], -1)
                #p (512, 4096 * 7 * 7)

                merged = merge @ p.clone().detach()

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
                merged =  merge @ p.clone().detach().reshape(sh[0], -1)
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


class WeightClustering:
    def __init__(self, n_clusters, n_features, normalize=False, use_kmeans=False):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.normalize  = normalize
        self.use_kmeans = use_kmeans
        
    def __call__(self, weight):
        km = AgglomerativeClustering(n_clusters=self.n_clusters)

        if self.use_kmeans is True:
            km = HKMeans(n_clusters=self.n_clusters, random_state=None, n_init=60,
                        n_jobs=-1, max_iter=10, verbose=True)

        X_scaled = weight.cpu().numpy()

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

        return merge.T, inertia


def self_merge_weight_clustering(perm_to_axes, params, max_ratio=0.5, threshold=0.1, hooks=None, approx_repair=False):
    perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}

    merges = dict()
    inertias = dict()
    ratios = dict()

    for p_name in perm_sizes.keys():
        print('merging block: "' + p_name + '"')
        ratio = max_ratio
        n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]

        # if p_name == "layer3.0.relu1":
        #     max_ratio = 0.35
        # elif p_name == "layer3.0.relu2":
        #     max_ratio = 0.97
        # elif p_name == "layer3.1.relu1":
        #     max_ratio = 0.35
        # elif p_name == "layer4.0.relu1":
        #     max_ratio = 0.35
        # elif p_name == "layer4.0.relu2":
        #     max_ratio = 0.35
        # elif p_name == "layer4.1.relu1":
        #     max_ratio = 0.35
        # else:
        #     max_ratio = 1.0
        
        if hooks is not None:
            distance = hooks[p_name].get_features().permute(1, 0, 2, 3)
            distance = distance.reshape(distance.shape[0], -1).detach().cpu()
        else:
            distance = self_weight_matching_p_name_raw_vec(perm_to_axes, params, p_name, n)
        
        merger = WeightClustering(int(distance.shape[0] * max_ratio), distance.shape[0], normalize=False)
        merge, inertia = merger(distance)
        merges[p_name] = merge
        inertias[p_name] = inertia

    for p_name in perm_sizes.keys():
        merge = merges[p_name]

        print("MERGE", p_name)
        if approx_repair is False:
            prm = merge_channel_p_name_clustering(perm_to_axes, params, p_name, merge)
        else:
            prm = merge_channel_p_name_clustering_approx_repair(perm_to_axes, params, p_name, merge)

        for wk, axis in perm_to_axes[p_name]:
            params[wk] = prm[wk]

    for p_name in perm_sizes.keys():
        wk, axis = perm_to_axes[p_name][0]
        sh0 = params[wk].shape
        print("SH1", p_name, sh0)

    new_perm_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    
    return params, new_perm_sizes
    
