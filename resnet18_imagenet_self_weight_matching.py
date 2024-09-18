from model.resnet_torch_flexi_new_mid import ResNet18, merge_channel_ResNet18

import argparse
import torch
import os
from torchvision import datasets, transforms

from tqdm import tqdm

from thop import profile

import torch.nn.functional as F

from utils.utils import ImageNet_test_transform

def test_activation_cluster(origin_model, checkpoint, dataloader, max_ratio, threshold, figure_path):
    input = torch.torch.randn(1, 3, 224, 224).cuda()
    origin_model.cuda()
    origin_model.eval()
    origin_flop, origin_param = profile(origin_model, inputs=(input,))
    model = merge_channel_ResNet18(checkpoint, max_ratio=max_ratio, threshold=threshold)
    model.cuda()
    model.eval()
    flop, param = profile(model, inputs=(input,))

    correct = 0
    total_num = 0
    total_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.cuda()
            y = y.cuda()
            logit = model(X)
            loss = F.cross_entropy(logit, y)
            correct += (logit.max(1)[1] == y).sum().item()
            total_num += y.size(0)
            total_loss += loss.item()
    print(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}")
    print(
        f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %")
    with open(os.path.join(figure_path, "self_merge_result_mid.txt"), 'a+') as f:
        f.write(f"max_ratio:{max_ratio}, threshold:{threshold}\n")
        f.write(f"model after adapt: acc:{correct/total_num * 100:.2f}%, avg loss:{total_loss / (i+1):.4f}\n")
        f.write(f"flop:{flop}/{origin_flop}, {flop / origin_flop * 100:.2f}%; param:{param}/{origin_param}, {param / origin_param * 100:.2f} %\n\n")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_param", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_ratio", default=1., type=float)
    parser.add_argument("--threshold", default=0.1, type=float)
    args = parser.parse_args()

    # load model
    model = ResNet18()
    checkpoint = torch.load(args.model_param)

    model.load_state_dict(checkpoint)
    model.cuda()

    testset = datasets.ImageFolder(root=os.path.join("/mnt/nas/dataset_share/imagenet", "val"), transform=ImageNet_test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64,
                                              shuffle=False, num_workers=8, pin_memory=True)
    figure_path = '/'.join(args.model_param.split('/')[:-1])
    test_activation_cluster(model, model.state_dict(), test_loader, args.max_ratio, args.threshold, figure_path)

if __name__ == "__main__":
  main()
