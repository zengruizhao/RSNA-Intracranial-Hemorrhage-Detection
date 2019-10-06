# -*- coding: utf-8 -*-
"""
@File    : resNetwork.py
@Time    : 2019/6/23 15:29
@Author  : Parker
@Email   : now_cherish@163.com
@Software: PyCharm
@Des     : 
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet101, densenet, inception_v3, mobilenet_v2
import torch.nn.functional as F
import pretrainedmodels


class Resnet18(nn.Module):
    def __init__(self, n_classes=6):
        super(Resnet18,self).__init__()

        src_net = resnet18(pretrained=True)
        modules = list(src_net.children())[:-2]
        modules[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(512, n_classes)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out,(1,1)).view(features.size(0), -1)
        out = nn.Sigmoid()(self.classifier(out))
        return out


class Resnet34(nn.Module):
    def __init__(self, n_classes=45):
        super(Resnet34, self).__init__()

        src_net = resnet34(pretrained=True)
        modules = list(src_net.children())[:-2]

        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(512, n_classes)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out,(1,1)).view(features.size(0), -1)
        out = self.classifier(out)
        # out = torch.sigmoid(out)
        return out


class Resnet101(nn.Module):
    def __init__(self, n_classes=45):
        super(Resnet101, self).__init__()

        src_net = resnet101(pretrained=True)
        modules = list(src_net.children())[:-2]

        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(2048, n_classes)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out,(1,1)).view(features.size(0), -1)
        out = self.classifier(out)
        # out = torch.sigmoid(out)
        return out


class Densenet121(nn.Module):
    def __init__(self, n_classes=45):
        super(Densenet121, self).__init__()

        src_net = densenet.densenet121(pretrained=False)
        # print(src_net)
        modules = list(src_net.children())[:-1]

        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(1024, n_classes)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        # out = torch.sigmoid(out)
        return out


class MobileNet(nn.Module):
    def __init__(self, n_classes=45):
        super(MobileNet,self).__init__()

        src_net = mobilenet_v2(pretrained=True)
        modules = list(src_net.children())[:-2]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.Linear(512, n_classes)
        )
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out,(1,1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class SEResNext50(nn.Module):
    def __init__(self, n_classes=6):
        super(SEResNext50, self).__init__()
        src_net = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        modules = list(src_net.children())[:-2]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(2048, n_classes)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = nn.Sigmoid()(self.classifier(out))

        return out


class Inceptionv4(nn.Module):
    def __init__(self, n_classes=6):
        super(Inceptionv4, self).__init__()
        src_net = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000,
                                                           pretrained='imagenet')
        modules = list(src_net.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(1536, n_classes)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = nn.Sigmoid()(self.classifier(out))

        return out



if __name__ == '__main__':
    net = Resnet18()
    print(net)
    # net = Densenet121()
    aa = torch.randn((5, 1, 512, 512))
    print(net(aa).size())

