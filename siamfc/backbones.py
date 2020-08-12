from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np

__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3']


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1_Test(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1_Test, self).__init__()

        rate=0.64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(96*rate), 11, 2),
            _BatchNorm2d(int(96*rate)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(96*rate), int(256*rate), 5, 1, groups=1),
            _BatchNorm2d(int(256*rate)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(256*rate), int(384*rate), 3, 1),
            _BatchNorm2d(int(384*rate)),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(int(384*rate), int(384*rate), 3, 1, groups=1),
            _BatchNorm2d(int(384*rate)),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(int(384*rate), 256, 3, 1, groups=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self, rate, ratio):
        super(AlexNetV1, self).__init__()
        self.rate = rate
        self.ratio = ratio
        self.device = torch.device('cuda:1')
        self.decay = torch.tensor(.9).cuda(self.device)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(96*rate), 11, 2),
            _BatchNorm2d(int(96*rate)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(96*rate), int(256*rate), 5, 1, groups=1),
            _BatchNorm2d(int(256*rate)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(256*rate), int(384*rate), 3, 1),
            _BatchNorm2d(int(384*rate)),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(int(384*rate), int(384*rate), 3, 1, groups=1),
            _BatchNorm2d(int(384*rate)),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(int(384*rate), 256, 3, 1, groups=1))

    def forward(self, x, epoch=0.0, batch_ind=0.0):
        if batch_ind * epoch < 1e-6:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            return x

        else:
            epm = np.exp( (epoch+batch_ind) * 3.0*0.0000001)

            x = self.conv1(x)
            xb, xc, xh, xw = x.size()
            att = torch.ones(1, xc, 1, 1).cuda(self.device)
            att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
            x = x.mul(att)

            x = self.conv2(x)
            xb, xc, xh, xw = x.size()
            att = torch.ones(1, xc, 1, 1).cuda(self.device)
            att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
            x = x.mul(att)

            x = self.conv3(x)
            xb, xc, xh, xw = x.size()
            att = torch.ones(1, xc, 1, 1).cuda(self.device)
            att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
            x = x.mul(att)

            x = self.conv4(x)
            xb, xc, xh, xw = x.size()
            att = torch.ones(1, xc, 1, 1).cuda(self.device)
            att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
            x = x.mul(att)

            x = self.conv5(x)
            '''
            xb, xc, xh, xw = x.size()
            att = torch.ones(1, xc, 1, 1).cuda(self.device)
            att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
            x = x.mul(att)
            '''
            return x


class AlexNetV2(_AlexNet):
    output_stride = 4

    def __init__(self):
        super(AlexNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2))


class AlexNetV3(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            _BatchNorm2d(512))
