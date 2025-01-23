import copy
import torch
import itertools
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn import preprocessing
from hkmeans import HKMeans


class NopMerge:
    def requires(self, arg):
        return False


def axes2perm_to_perm2axes(axes_to_perm):
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return perm_to_axes


class WeightClustering:
    def __init__(self, n_clusters, n_features, normalize=False, use_kmeans=True):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.normalize  = normalize
        self.use_kmeans = use_kmeans
        
    def __call__(self, weight):
        km = AgglomerativeClustering(n_clusters=self.n_clusters)

        if self.use_kmeans is True:
            km = HKMeans(n_clusters=self.n_clusters, random_state=None, n_init=10,
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


def concat_weights(perm_to_axes, params, p_name, n, approx_repair=False):
  A = None
  for wk, axis in perm_to_axes[p_name]:
    w_a = params[wk]

    if "running" in wk or "identity_transform" in wk:
        continue

    if approx_repair is True:
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


def merge_channel_p_name_clustering(perm_to_axes, params, p_name, merge, custom_merger=None):
    params = copy.deepcopy(params)
    for wk, axis in perm_to_axes[p_name]:
        assert axis in (0, 1)
        if axis == 0:
            p = params[wk].detach().clone()
            if len(p.shape) == 1:
                params[wk] = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p 
            else:
                sh = p.shape
                merged = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p.clone().detach().reshape(sh[0], -1)
                merged = merged.reshape(merge.shape[0], *sh[1:])
                params[wk] = merged
        else:
            p = params[wk].detach().clone()

            if custom_merger.requires(wk):
                params[wk] = custom_merger.merge(wk, p, merge)
            else:
                if len(p.shape) == 2:
                    p = p.permute(1, 0)
                else:
                    p = p.permute(1, 0, 2, 3)
            
                sh = p.shape
                print(wk, merge.shape, p.shape)
                merged =  merge @ p.clone().detach().reshape(sh[0], -1)
                merged = merged.reshape(merge.shape[0], *sh[1:])

                if len(p.shape) == 2:
                    merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0)
                else:
                    merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0, 2, 3)

                params[wk] = merged

    return params


def merge_channel_approx_repair(perm_to_axes, params, p_name, merge, custom_merger=None):
    params = copy.deepcopy(params)
    n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]
    w = concat_weights(perm_to_axes, params, p_name, n)

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
                    n_c        = torch.sum(merge, axis=1)
                    params[wk] = params[wk] * torch.sqrt(n_c / (1 + (n_c - 1) * avg_corr))
            else:
                sh         = p.shape
                merged     = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p.clone().detach().reshape(sh[0], -1)
                merged     = merged.reshape(merge.shape[0], *sh[1:])
                params[wk] = merged
        else:
            p = params[wk].detach().clone()

            if custom_merger.requires(wk):
                params[wk] = custom_merger.merge(wk, p, merge)
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


def self_merge_weight_clustering(perm_to_axes, params, max_ratio=0.5, threshold=0.1, hooks=None, approx_repair=False, merge_layer=None, custom_merger=None):
    merge_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    merges      = dict()
    inertias    = dict()

    if custom_merger is None:
        custom_merger = NopMerge()

    for p_name in merge_sizes.keys():
        print('Compressing block: "' + p_name + '"')

        n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]

        ratio = max_ratio
        if merge_layer is not None:
            if p_name != merge_layer:
                ratio = 1.0
        
        if hooks is not None:
            features =  hooks[p_name].get_features()

            if len(features.shape) == 2:
                distance = features.permute(1, 0)
            elif len(features.shape) == 4:
                distance = features.permute(1, 0, 2, 3)

            distance = distance.reshape(distance.shape[0], -1).detach().cpu()
        else:
            distance = concat_weights(perm_to_axes, params, p_name, n)
        
        merger = WeightClustering(int(distance.shape[0] * ratio), distance.shape[0], normalize=approx_repair)
        merge, inertia = merger(distance)
        merges[p_name] = merge
        inertias[p_name] = inertia

    for p_name in merge_sizes.keys():
        print('Merging block: "' + p_name + '"')

        merge = merges[p_name]

        if approx_repair is False:
            prm = merge_channel_p_name_clustering(perm_to_axes, params, p_name, merge, custom_merger=custom_merger)
        else:
            prm = merge_channel_approx_repair(perm_to_axes, params, p_name, merge, custom_merger=custom_merger)

        for wk, axis in perm_to_axes[p_name]:
            params[wk] = prm[wk]

    new_merge_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    
    return params, new_merge_sizes
    

def self_merge_weight_merging(perm_to_axes, params_a, params_b, hooks=None, approx_repair=False, merge_layer=None, custom_merger=None):
    merge_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    merges      = dict()
    inertias    = dict()
    max_ratio   = 0.5

    if custom_merger is None:
        custom_merger = NopMerge()
    
    for p_name in merge_sizes.keys():
        print('merging block: "' + p_name + '"')
        n = params_a[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]

        ratio = max_ratio
        if merge_layer is not None:
            if p_name != merge_layer:
                ratio = 1.0

        if hooks is not None:
            features =  hooks[p_name].get_features()

            if len(features.shape) == 2:
                distance = features.permute(1, 0)
            elif len(features.shape) == 4:
                distance = features.permute(1, 0, 2, 3)

            distance = distance.reshape(distance.shape[0], -1).detach().cpu()
        else:
            distance_a = concat_weights(perm_to_axes, params_a, p_name, n)
            distance_b = concat_weights(perm_to_axes, params_b, p_name, n)
            distance = torch.vstack((distance_a, distance_b))
        
        merger = WeightClustering(int(distance.shape[0] * ratio), distance.shape[0], normalize=approx_repair)
        merge, inertia = merger(distance)
        merges[p_name] = merge
        inertias[p_name] = inertia

    params = copy.deepcopy(params_a)

    for key in params.keys():
        if "num_batches_tracked" in key:
            continue
        
        if len(params[key].shape) == 4:
            params[key] = torch.zeros(
                (
                    2 * params_a[key].shape[0], 2 * params_a[key].shape[1], params_a[key].shape[2], params_a[key].shape[3]
                )
            ).cuda()
            params[key][:params_a[key].shape[0], :params_a[key].shape[1], :, :] = params_a[key].detach().clone()
            params[key][params_a[key].shape[0]:, params_a[key].shape[1]:, :, :] = params_b[key].detach().clone()

        if len(params[key].shape) == 2:
            params[key] = torch.zeros(
                (
                    2 * params_a[key].shape[0], 2 * params_a[key].shape[1]
                )
            ).cuda()
            params[key][:params_a[key].shape[0], :params_a[key].shape[1]] = params_a[key].detach().clone()
            params[key][params_a[key].shape[0]:, params_a[key].shape[1]:] = params_b[key].detach().clone()
        
        if len(params[key].shape) == 1:
            params[key] = torch.zeros(
                (
                    2 * params_a[key].shape[0]
                )
            ).cuda()
            params[key][:params_a[key].shape[0]] = params_a[key].detach().clone()
            params[key][params_a[key].shape[0]:] = params_b[key].detach().clone()

    for p_name in merges.keys():
        merge = merges[p_name]
        
        if approx_repair is False:
            prm = merge_channel_p_name_clustering(perm_to_axes, params, p_name, merge, custom_merger=custom_merger)
        else:
            prm = merge_channel_approx_repair(perm_to_axes, params, p_name, merge, custom_merger=custom_merger)

        for wk, axis in perm_to_axes[p_name]:
            params[wk] = prm[wk]

    new_merge_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    
    return params, new_merge_sizes
