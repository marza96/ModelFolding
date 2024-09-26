import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import numpy as np


def reset_bn_stats(model, loader, device, epochs=1):
    num_data = 0
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None # use simple average
            m.reset_running_stats()

    # run a single train epoch
    model.train()
    for _ in range(epochs):
        with torch.no_grad():
            for images, _ in loader:
                output = model(images.to(device))
                num_data+=len(images)
                if num_data>=1000:
                    print("Enough data for REPAIR")
                    break
                    
    model.eval()

    return model


def local_structured_prune_model(model, pruning_rate=0.25, n=1, save_path=None):
    pruned_channels = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if n == 1:
                     importance = torch.sum(torch.abs(module.weight), dim=tuple(range(1, module.weight.dim())))
            elif n == 2:
                importance = torch.sqrt(torch.sum(module.weight ** 2, dim=tuple(range(1, module.weight.dim()))))
            else:
                importance = torch.sum(torch.abs(module.weight) ** n, dim=tuple(range(1, module.weight.dim()))) ** (1/n)
            if save_path is not None:
                importance_module_path = f'{save_path}/{name}_importance.npy'
                print(f'Save importance in {importance_module_path}')
                np.save(importance_module_path, importance.detach().cpu().numpy())
        
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=n, dim=0)
            # prune.ln_structured(module, name='bias', amount=pruning_rate, n=n, dim=0)
            prune.remove(module, 'weight')
            # prune.remove(module, 'bias')

            sparsity = 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
            print(f"Sparsity in {name}.weight: {sparsity:.2f}%")
            
            if isinstance(module, nn.Conv2d):
                pruned_channels[name] = [i for i, w in enumerate(module.weight.detach().cpu().numpy()) if not w.any()]
            elif isinstance(module, nn.Linear):
                pruned_channels[name] = [i for i, w in enumerate(module.weight.detach().cpu().numpy()) if not w.any()]
            
            # print(f"Pruned channels in {name}: {pruned_channels[name]}")
            last_pruned_channels = [i for i, w in enumerate(module.weight.detach().cpu().numpy()) if not w.any()]

        elif isinstance(module, nn.BatchNorm2d):
            if last_pruned_channels is not None:
                prune_mask = torch.ones(module.weight.data.shape).to(device=module.weight.data.device)
                prune_mask[last_pruned_channels] = 0
                module.weight.data.mul_(prune_mask)
                module.bias.data.mul_(prune_mask)
                module.running_mean.data.mul_(prune_mask)
                module.running_var.data.mul_(prune_mask)
                pruned_channels[name] = [i for i, w in enumerate(module.weight.detach().cpu().numpy()) if not w.any()]
            last_pruned_channels = None

    return pruned_channels