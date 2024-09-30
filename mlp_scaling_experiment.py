import argparse
import torch
import wandb
import copy


from model.mlp import MLP, merge_channel_mlp_clustering, merge_channel_mlp_clustering_approx_repair
from utils.utils import DI_REPAIR, NO_REPAIR, REPAIR, DF_REPAIR, eval_model, fuse_bnorms_mlp
from utils.datasets import get_cifar10
from thop import profile
from tqdm import tqdm



def test_merge(origin_model, checkpoint, dataloader, train_loader, max_ratio, method, repair, di_samples_path, eval=True, measure_variance=False):
    input = torch.torch.randn(1, 3, 32, 32).cuda()
    origin_model.cuda()
    origin_model.eval()

    origin_flop, origin_param = profile(origin_model, inputs=(input,))

    model = method(copy.deepcopy(origin_model), checkpoint, max_ratio=max_ratio, hooks=None)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    if repair != NO_REPAIR and repair != DF_REPAIR:
        if repair == DI_REPAIR: 
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            model(torch.load(di_samples_path).to("cuda"))
            model.eval()

        elif repair == REPAIR:
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.reset_running_stats()
                    module.momentum = None

            model.train()
            for x, _ in tqdm(train_loader):
                model(x.to("cuda"))
                break
                
            model.eval()
    
    if eval is True:
        acc, loss = eval_model(model, dataloader)
        print(f"model after adapt: acc:{acc * 100:.2f}%, avg loss:{loss:.4f}")
        print(f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    
        return model, acc, param / origin_param

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--repair", type=str, default="NO_REPAIR", help="")
    parser.add_argument("--proj_name", type=str, help="", default="Folding MLP cifar10")
    parser.add_argument("--exp_name", type=str, help="", default="WM REPAIR")
    parser.add_argument("--di_samples_path", type=str, default="mlp3x_cifar10.pt")
    args = parser.parse_args()

    model = MLP(wider_factor=args.width)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.cuda()

    test_loader = get_cifar10(train=False)
    train_loader = get_cifar10(train=True)

    method = merge_channel_mlp_clustering
    if args.repair == DF_REPAIR:
        fuse_bnorms_mlp(model)
        method = merge_channel_mlp_clustering_approx_repair

    desc = {"experiment": args.exp_name}
    wandb.init(
        project=args.proj_name,
        config=desc,
        name=args.exp_name
    )

    for ratio in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]: #, 0.65, 0.75, 0.85, 0.95]:
        new_model, acc, sparsity = test_merge(copy.deepcopy(model), copy.deepcopy(model).state_dict(), test_loader, train_loader, ratio, method, args.repair, args.di_samples_path)
        wandb.log({"test acc": acc})
        wandb.log({"sparsity": 1.0 - sparsity})


if __name__ == "__main__":
  main()