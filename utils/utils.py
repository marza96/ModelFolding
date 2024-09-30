import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F


NO_REPAIR = "NO_REPAIR"
DF_REPAIR = "DF_REPAIR"
DI_REPAIR = "DI_REPAIR"
REPAIR    = "REPAIR"


class AvgLayerStatisticsHook:
    def __init__(self, conv=False):
        self.conv = conv
        self.bnorm = None

    def __call__(self, module, input, output):
        if self.bnorm is None:
            if self.conv is True:
                self.bnorm = torch.nn.BatchNorm2d(output.shape[1]).to("cuda")
            else:
                self.bnorm = torch.nn.BatchNorm1d(output.shape[1]).to("cuda")

            self.bnorm.train()
            self.bnorm.momentum = None
        
        self.bnorm(output)

    def get_stats(self):
        return self.bnorm.running_var.mean()


def fuse_bnorms_mlp(model):
    alpha = torch.diag(model.bn1.weight.data.clone().detach())
    beta = model.bn1.bias.data.clone().detach()
    rm = model.bn1.running_mean.data.clone().detach()
    rv = torch.sqrt(model.bn1.running_var.data.clone().detach())
    rv = torch.diag(1.0 / rv)
    model.fc1.weight.data = alpha @ rv @ model.fc1.weight.data.clone().detach()
    model.fc1.bias.data = alpha @ rv @ ( model.fc1.bias.data.clone().detach()  - rm) + beta
    model.bn1.running_var.data = torch.ones_like(model.bn1.weight.data)
    model.bn1.running_mean.data = torch.zeros_like(model.bn1.bias.data)

    alpha = torch.diag(model.bn2.weight.data.clone().detach())
    beta = model.bn2.bias.data.clone().detach()
    rm = model.bn2.running_mean.data.clone().detach()
    rv = torch.sqrt(model.bn2.running_var.data.clone().detach())
    rv = torch.diag(1.0 / rv)
    model.fc2.weight.data = alpha @ rv @ model.fc2.weight.data.clone().detach()
    model.fc2.bias.data = alpha @ rv @ ( model.fc2.bias.data.clone().detach()  - rm) + beta
    model.bn2.running_var.data = torch.ones_like(model.bn2.weight.data)
    model.bn2.running_mean.data = torch.zeros_like(model.bn2.bias.data)

    alpha = torch.diag(model.bn3.weight.data.clone().detach())
    beta = model.bn3.bias.data.clone().detach()
    rm = model.bn3.running_mean.data.clone().detach()
    rv = torch.sqrt(model.bn3.running_var.data.clone().detach())
    rv = torch.diag(1.0 / rv)
    model.fc3.weight.data = alpha @ rv @ model.fc3.weight.data.clone().detach()
    model.fc3.bias.data = alpha @ rv @ ( model.fc3.bias.data.clone().detach()  - rm) + beta
    model.bn3.running_var.data = torch.ones_like(model.bn3.weight.data)
    model.bn3.running_mean.data = torch.zeros_like(model.bn3.bias.data)


def fuse_bnorms_arbitrary(block, block_len):
    for i in range(block_len):
        alpha = block[i].bn1.weight.data.clone().detach()
        beta = block[i].bn1.bias.data.clone().detach()
        block[i].bn1.weight.data = torch.ones_like(block[i].bn1.weight.data)
        block[i].bn1.bias.data = torch.zeros_like(block[i].bn1.bias.data)

        block[i].conv1 = ConvBnormFuse(
            block[i].conv1,
            block[i].bn1
        ).fused
        block[i].bn1.weight.data = alpha
        block[i].bn1.bias.data = beta
        block[i].bn1.running_mean.data = torch.zeros_like(block[i].bn1.running_mean.data)
        block[i].bn1.running_var.data = torch.ones_like(block[i].bn1.running_var.data)


        alpha = block[i].bn2.weight.data.clone().detach()
        beta = block[i].bn2.bias.data.clone().detach()
        block[i].bn2.weight.data = torch.ones_like(block[i].bn2.weight.data)
        block[i].bn2.bias.data = torch.zeros_like(block[i].bn2.bias.data)

        block[i].conv2 = ConvBnormFuse(
            block[i].conv2,
            block[i].bn2
        ).fused
        block[i].bn2.weight.data = alpha
        block[i].bn2.bias.data = beta
        block[i].bn2.running_mean.data = torch.zeros_like(block[i].bn2.running_mean.data)
        block[i].bn2.running_var.data = torch.ones_like(block[i].bn2.running_var.data)


        alpha = block[i].bn3.weight.data.clone().detach()
        beta = block[i].bn3.bias.data.clone().detach()
        block[i].bn3.weight.data = torch.ones_like(block[i].bn3.weight.data)
        block[i].bn3.bias.data = torch.zeros_like(block[i].bn3.bias.data)

        block[i].conv3 = ConvBnormFuse(
            block[i].conv3,
            block[i].bn3
        ).fused
        block[i].bn3.weight.data = alpha
        block[i].bn3.bias.data = beta
        block[i].bn3.running_mean.data = torch.zeros_like(block[i].bn3.running_mean.data)
        block[i].bn3.running_var.data = torch.ones_like(block[i].bn3.running_var.data)

        if len(block[i].shortcut) == 2:
            alpha = block[i].shortcut[1].weight.data.clone().detach()
            beta = block[i].shortcut[1].bias.data.clone().detach()
            block[i].shortcut[1].weight.data = torch.ones_like(block[i].shortcut[1].weight.data)
            block[i].shortcut[1].bias.data = torch.zeros_like(block[i].shortcut[1].bias.data)
            block[i].shortcut[0] = ConvBnormFuse(
                block[i].shortcut[0],
                block[i].shortcut[1]
            ).fused
            block[i].shortcut[1].weight.data = alpha
            block[i].shortcut[1].bias.data = beta
            block[i].shortcut[1].running_mean.data = torch.zeros_like(block[i].shortcut[1].running_mean.data)
            block[i].shortcut[1].running_var.data = torch.ones_like(block[i].shortcut[1].running_var.data)


def fuse_bnorms_arbitrary_override(block, block_len):
    for i in range(block_len):
        alpha = block[i].bn1.weight.data.clone().detach()
        beta = block[i].bn1.bias.data.clone().detach()
        block[i].bn1.weight.data = torch.ones_like(block[i].bn1.weight.data)
        block[i].bn1.bias.data = torch.zeros_like(block[i].bn1.bias.data)

        block[i].conv1 = ConvBnormFuse(
            block[i].conv1,
            block[i].bn1
        ).fused
        block[i].bn1.weight.data = alpha
        block[i].bn1.bias.data = beta
        block[i].bn1.running_mean.data = torch.zeros_like(block[i].bn1.running_mean.data)
        block[i].bn1.running_var.data = torch.ones_like(block[i].bn1.running_var.data)


        alpha = block[i].bn2.weight.data.clone().detach()
        beta = block[i].bn2.bias.data.clone().detach()
        block[i].bn2.weight.data = torch.ones_like(block[i].bn2.weight.data)
        block[i].bn2.bias.data = torch.zeros_like(block[i].bn2.bias.data)

        block[i].conv2 = ConvBnormFuse(
            block[i].conv2,
            block[i].bn2
        ).fused
        block[i].bn2.weight.data = alpha
        block[i].bn2.bias.data = beta
        block[i].bn2.running_mean.data = torch.zeros_like(block[i].bn2.running_mean.data)
        block[i].bn2.running_var.data = torch.ones_like(block[i].bn2.running_var.data)


        alpha = block[i].bn3.weight.data.clone().detach()
        beta = block[i].bn3.bias.data.clone().detach()
        block[i].bn3.weight.data = torch.ones_like(block[i].bn3.weight.data)
        block[i].bn3.bias.data = torch.zeros_like(block[i].bn3.bias.data)

        block[i].conv3 = ConvBnormFuse(
            block[i].conv3,
            block[i].bn3
        ).fused
        block[i].bn3.weight.data = alpha
        block[i].bn3.bias.data = beta
        block[i].bn3.running_mean.data = torch.zeros_like(block[i].bn3.running_mean.data)
        block[i].bn3.running_var.data = torch.ones_like(block[i].bn3.running_var.data)

        if block[i].downsample is not None:
            alpha = block[i].downsample[1].weight.data.clone().detach()
            beta = block[i].downsample[1].bias.data.clone().detach()
            block[i].downsample[1].weight.data = torch.ones_like(block[i].downsample[1].weight.data)
            block[i].downsample[1].bias.data = torch.zeros_like(block[i].downsample[1].bias.data)
            block[i].downsample[0] = ConvBnormFuse(
                block[i].downsample[0],
                block[i].downsample[1]
            ).fused
            block[i].downsample[1].weight.data = alpha
            block[i].downsample[1].bias.data = beta
            block[i].downsample[1].running_mean.data = torch.zeros_like(block[i].downsample[1].running_mean.data)
            block[i].downsample[1].running_var.data = torch.ones_like(block[i].downsample[1].running_var.data)


def fuse_bnorms_arbitrary_resnet(model, blocks, override=True):
    alpha = model.bn1.weight.data.clone().detach()
    beta = model.bn1.bias.data.clone().detach()
    model.bn1.weight.data = torch.ones_like(model.bn1.weight.data)
    model.bn1.bias.data = torch.zeros_like(model.bn1.bias.data)

    model.conv1 = ConvBnormFuse(
        model.conv1,
        model.bn1
    ).fused
    model.bn1.weight.data = alpha
    model.bn1.bias.data = beta
    model.bn1.running_mean.data = torch.zeros_like(model.bn1.running_mean.data)
    model.bn1.running_var.data = torch.ones_like(model.bn1.running_var.data)

    if override is True:
        fuse_bnorms_arbitrary_override(model.layer1, blocks[0])
        fuse_bnorms_arbitrary_override(model.layer2, blocks[1])
        fuse_bnorms_arbitrary_override(model.layer3, blocks[2])
        fuse_bnorms_arbitrary_override(model.layer4, blocks[3])

        return
    
    fuse_bnorms_arbitrary(model.layer1, blocks[0])
    fuse_bnorms_arbitrary(model.layer2, blocks[1])
    fuse_bnorms_arbitrary(model.layer3, blocks[2])
    fuse_bnorms_arbitrary(model.layer4, blocks[3])
    

def fuse_bnorms_vgg(model):
    for i in range(len(model.features)):
        if isinstance(model.features[i], torch.nn.Conv2d):
            print(type(model.features[i]), model.features[i + 1])

            alpha = model.features[i + 1].weight.data.clone().detach()
            beta = model.features[i + 1].bias.data.clone().detach()
            model.features[i + 1].weight.data = torch.ones_like(model.features[i + 1].weight.data)
            model.features[i + 1].bias.data = torch.zeros_like(model.features[i + 1].bias.data)

            model.features[i] = ConvBnormFuse(
                model.features[i],
                model.features[i + 1]
            ).fused
            model.features[i + 1].weight.data = alpha
            model.features[i + 1].bias.data = beta
            model.features[i + 1].running_mean.data = torch.zeros_like(model.features[i + 1].running_mean.data)
            model.features[i + 1].running_var.data = torch.ones_like(model.features[i + 1].running_var.data)


def fuse_bnorms_resnet18(model, override=False):
    alpha = model.bn1.weight.data.clone().detach()
    beta = model.bn1.bias.data.clone().detach()
    model.bn1.weight.data = torch.ones_like(model.bn1.weight.data)
    model.bn1.bias.data = torch.zeros_like(model.bn1.bias.data)

    model.conv1 = ConvBnormFuse(
        model.conv1,
        model.bn1
    ).fused
    model.bn1.weight.data = alpha
    model.bn1.bias.data = beta
    model.bn1.running_mean.data = torch.zeros_like(model.bn1.running_mean.data)
    model.bn1.running_var.data = torch.ones_like(model.bn1.running_var.data)

    if override is True:
        fuse_bnorms_basic_block(model.layer1)
        fuse_bnorms_basic_block(model.layer2)
        fuse_bnorms_basic_block(model.layer3)
        fuse_bnorms_basic_block(model.layer4)
        
        return
    
    fuse_bnorms_basic_block_override(model.layer1)
    fuse_bnorms_basic_block_override(model.layer2)
    fuse_bnorms_basic_block_override(model.layer3)
    fuse_bnorms_basic_block_override(model.layer4)


def fuse_bnorms_basic_block(block):
    for i in range(2):
        alpha = block[i].bn1.weight.data.clone().detach()
        beta = block[i].bn1.bias.data.clone().detach()
        block[i].bn1.weight.data = torch.ones_like(block[i].bn1.weight.data)
        block[i].bn1.bias.data = torch.zeros_like(block[i].bn1.bias.data)

        block[i].conv1 = ConvBnormFuse(
            block[i].conv1,
            block[i].bn1
        ).fused
        block[i].bn1.weight.data = alpha
        block[i].bn1.bias.data = beta
        block[i].bn1.running_mean.data = torch.zeros_like(block[i].bn1.running_mean.data)
        block[i].bn1.running_var.data = torch.ones_like(block[i].bn1.running_var.data)

        alpha = block[i].bn2.weight.data.clone().detach()
        beta = block[i].bn2.bias.data.clone().detach()
        block[i].bn2.weight.data = torch.ones_like(block[i].bn2.weight.data)
        block[i].bn2.bias.data = torch.zeros_like(block[i].bn2.bias.data)

        block[i].conv2 = ConvBnormFuse(
            block[i].conv2,
            block[i].bn2
        ).fused
        block[i].bn2.weight.data = alpha
        block[i].bn2.bias.data = beta
        block[i].bn2.running_mean.data = torch.zeros_like(block[i].bn2.running_mean.data)
        block[i].bn2.running_var.data = torch.ones_like(block[i].bn2.running_var.data)

        if len(block[i].shortcut) == 2:
            alpha = block[i].shortcut[1].weight.data.clone().detach()
            beta = block[i].shortcut[1].bias.data.clone().detach()
            block[i].shortcut[1].weight.data = torch.ones_like(block[i].shortcut[1].weight.data)
            block[i].shortcut[1].bias.data = torch.zeros_like(block[i].shortcut[1].bias.data)
            block[i].shortcut[0] = ConvBnormFuse(
                block[i].shortcut[0],
                block[i].shortcut[1]
            ).fused
            block[i].shortcut[1].weight.data = alpha
            block[i].shortcut[1].bias.data = beta
            block[i].shortcut[1].running_mean.data = torch.zeros_like(block[i].shortcut[1].running_mean.data)
            block[i].shortcut[1].running_var.data = torch.ones_like(block[i].shortcut[1].running_var.data)


def fuse_bnorms_basic_block_override(block):
    for i in range(2):
        alpha = block[i].bn1.weight.data.clone().detach()
        beta = block[i].bn1.bias.data.clone().detach()
        block[i].bn1.weight.data = torch.ones_like(block[i].bn1.weight.data)
        block[i].bn1.bias.data = torch.zeros_like(block[i].bn1.bias.data)

        block[i].conv1 = ConvBnormFuse(
            block[i].conv1,
            block[i].bn1
        ).fused
        block[i].bn1.weight.data = alpha
        block[i].bn1.bias.data = beta
        block[i].bn1.running_mean.data = torch.zeros_like(block[i].bn1.running_mean.data)
        block[i].bn1.running_var.data = torch.ones_like(block[i].bn1.running_var.data)

        alpha = block[i].bn2.weight.data.clone().detach()
        beta = block[i].bn2.bias.data.clone().detach()
        block[i].bn2.weight.data = torch.ones_like(block[i].bn2.weight.data)
        block[i].bn2.bias.data = torch.zeros_like(block[i].bn2.bias.data)

        block[i].conv2 = ConvBnormFuse(
            block[i].conv2,
            block[i].bn2
        ).fused
        block[i].bn2.weight.data = alpha
        block[i].bn2.bias.data = beta
        block[i].bn2.running_mean.data = torch.zeros_like(block[i].bn2.running_mean.data)
        block[i].bn2.running_var.data = torch.ones_like(block[i].bn2.running_var.data)

        if block[i].downsample is not None:
            alpha = block[i].downsample[1].weight.data.clone().detach()
            beta = block[i].downsample[1].bias.data.clone().detach()
            block[i].downsample[1].weight.data = torch.ones_like(block[i].downsample[1].weight.data)
            block[i].downsample[1].bias.data = torch.zeros_like(block[i].downsample[1].bias.data)
            block[i].downsample[0] = ConvBnormFuse(
                block[i].downsample[0],
                block[i].downsample[1]
            ).fused
            block[i].downsample[1].weight.data = alpha
            block[i].downsample[1].bias.data = beta
            block[i].downsample[1].running_mean.data = torch.zeros_like(block[i].downsample[1].running_mean.data)
            block[i].downsample[1].running_var.data = torch.ones_like(block[i].downsample[1].running_var.data)


def eval_model(model, dataloader):
    model.eval()
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

    return correct / total_num, total_loss / total_num


def load_model(model, i, override=True):
    sd = torch.load(i, map_location=torch.device('cpu'))
    new_sd = copy.deepcopy(sd)

    if override is True:
        for key, value in sd.items():
            if "downsample" in key:
                new_key = key.replace("downsample", "shortcut")
                new_sd[new_key] = value
                new_sd.pop(key)

            if "fc" in key:
                new_key = key.replace("fc", "linear")
                new_sd[new_key] = value
                new_sd.pop(key)

    model.load_state_dict(new_sd)


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

