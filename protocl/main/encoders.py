import torch
import torch.nn as nn
import torch.nn.functional as F
from protocl.main.utils import Flatten, conv_block, final_conv_block

def get_encoder(config):
    hid_dim = config['model.hid_dim']
    x_dim = config['model.x_dim']
    phi_dim = config['model.phi_dim']
    dropout_prob = config['train.dropout_prob']

    # CLASSIFICATION

    if config['data.dataset'] in ['SplitMNIST', 'SplitFMNIST']:
        activation = nn.ReLU()
        encoder = nn.Sequential(
            nn.Linear(x_dim, hid_dim),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hid_dim, phi_dim),
            nn.Dropout(p=dropout_prob),
        )
    elif config['data.dataset'] == 'MNIST':
        activation = nn.ReLU()
        encoder = nn.Sequential(
            nn.Linear(x_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, phi_dim),
            activation,
        )

    elif config['data.dataset'] in ['SplitCIFAR10', 'SplitCIFAR100', 'CIFAR100']:
        if config["model.large_model"]:
            encoder = resnet18(phi_dim)
        else:
            encoder = CifarNet(3, phi_dim)

    elif config['data.dataset'] == 'MiniImageNet':
        encoder = nn.Sequential(
            conv_block(3, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            # final_conv_block(hid_dim, hid_dim),
            final_conv_block(hid_dim, phi_dim),
            Flatten()
        )

    # elif config['data.dataset'] == 'PermutedMNIST':
    #     encoder = nn.Sequential(
    #         conv_block(1, hid_dim),
    #         conv_block(hid_dim, hid_dim),
    #         conv_block(hid_dim, hid_dim),
    #         final_conv_block(hid_dim, hid_dim),
    #         Flatten()
    #     )

    else:
        raise ValueError("data.dataset not understood")

    return encoder

class CifarNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(type(self), self).__init__()
        print("CIFARNet from FROMP")
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )
        self.linear_block = nn.Sequential(
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.out_block = nn.Linear(512, out_channels)


    def weight_init(self):
        nn.init.constant_(self.out_block.weight, 0)
        nn.init.constant_(self.out_block.bias, 0)

    def forward(self, x):
        o = self.conv_block(x)
        o = torch.flatten(o, 1)
        o = self.linear_block(o)
        o = self.out_block(o)
        return o


## https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, phi=128):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, phi)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(phi=128, dropout=False, per_img_std=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], phi)
    return model


def resnet34(phi=128, dropout=False, per_img_std=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], phi)
    return model


def resnet50(phi=128, dropout=False, per_img_std=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], phi)
    return model


def resnet101(phi=128, dropout=False, per_img_std=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], phi)
    return model


def resnet152(phi=128, dropout=False, per_img_std=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], phi)
    return model