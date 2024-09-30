# Running the code
## Approximate REPAIR experiments
To run an experiment involving approximate (data-free) REPAIR type the following:
```
python <experiment_name>_approx_repair.py --checkpoint <path_to_weights> --proj_name <wandb project name> --exp_name <wandb experiment name>
```
For example:
```
python resnet18_cifar10_weight_clustering_approx_repair.py --checkpoint ./resnet18_cifar10.pt --proj_name foo --exp_name bar
```
## REPAIR experiments 
To run an experiment involving REPAIR type the following:
```
python <experiment_name>.py --checkpoint <path_to_weights> --proj_name <wandb project name> --exp_name <wandb experiment name> --repair "REPAIR"
```
For example:
```
python resnet18_cifar10_weight_clustering_approx_repair.py --checkpoint ./resnet18_cifar10.pt --proj_name foo --exp_name bar --repair "REPAIR"
```
## NO REPAIR experiments 
To run the experiments without REPAIR type the following:
```
python <experiment_name>.py --checkpoint <path_to_weights> --proj_name <wandb project name> --exp_name <wandb experiment name> --repair "REPAIR"
```
For example:
```
python resnet18_cifar10_weight_clustering_approx_repair.py --checkpoint ./resnet18_cifar10.pt --proj_name foo --exp_name bar --repair "NO_REPAIR"
```
## Deep Inversion REPAIR experiments 
### **Important Note:**
To generate the Deep Inversion synthetic samples for Deep Inversion REPAIR please refer to [Link text](https://github.com/NVlabs/DeepInversion)

To run the experiments without REPAIR type the following:
```
python <experiment_name>.py --checkpoint <path_to_weights> --proj_name <wandb project name> --exp_name <wandb experiment name> --repair "DI_REPAIR" --di_samples_path <path_to_deep_inversion_images>
```
where <path_to_deep_inversion_images> denotes a serialized tensor containing the Deep Inversion synthetic samples extracted from the uncompressed model. For example:
```
python resnet18_cifar10_weight_clustering_approx_repair.py --checkpoint ./resnet18_cifar10.pt --proj_name foo --exp_name bar --repair "DI_REPAIR" --di_samples_path ./resnet18_di_samples.pt
```

