import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from typing import Optional
from torch.optim import optimizer
from collections import OrderedDict

class LabelSmoothLoss(nn.Module):
    def __init__(self, n_class, alpha) -> None:
        super(LabelSmoothLoss, self).__init__()
        self.n_class = n_class
        self.lb_pos = 1.0 - alpha
        self.lb_neg = alpha / (n_class - 1)

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            label_onehot = torch.empty_like(pred).fill_(self.lb_neg).scatter_(
                dim=1, index=target.data.unsqueeze(1), value=self.lb_pos)
        return torch.mean(-torch.sum(label_onehot * pred, dim=-1))


class Guassian(object):
    def __init__(self, am) -> None:
        super().__init__()
        self.am = am

    def __call__(self, input):
        input = input + self.am * torch.randn(input.size())
        return input


class ChannelWiseGuassian(object):
    def __init__(self, max_am) -> None:
        super().__init__()
        self.max_am = max_am

    def __call__(self, input):
        input_am = torch.mean(torch.std(input, dim=-1))
        shape = input.shape
        am = self.max_am * torch.rand(size=(1, shape[1], 1)) * input_am
        input += am * torch.randn(size=shape)

        return input


class ChannelAMChange(object):
    def __init__(self, max_ratio=1.5):
        super().__init__()
        self.max_ratio = max_ratio

    def __call__(self, input):
        shape = input.shape
        input = self.max_ratio * torch.rand(size=(1, shape[1], 1)) * input
        return input


class Sampling(object):
    def __init__(self, scale_factor) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, input):
        shape = list(input.shape)
        sf = torch.rand(1) * self.scale_factor + 0.5
        input = F.interpolate(input,
                              size=[int(shape[-1] * sf)],
                              mode='nearest')
        input = F.interpolate(input, size=shape[-1], mode='nearest')
        return input


class Shifting(object):
    def __init__(self, shift_scale):
        self.shift_scale = shift_scale

    def __call__(self, input):
        shift = np.random.randint(int(self.shift_scale * 128))
        direction = 1 if np.random.random(1) >= 0.5 else -1
        shift = direction * shift

        input = torch.roll(input, shift)

        return input


class ChannelShuffle(object):
    def __init__(self, shuffle_rate):
        self.shuffle_rate = shuffle_rate

    def __call__(self, input):
        num = int(self.shuffle_rate * input.shape[-2])
        idx = np.random.permutation(np.arange(input.shape[-2]))
        idx = idx[:num]
        shuffle_idx = np.random.permutation(idx)
        input[:, idx, :] = input[:, shuffle_idx, :]

        return input


class RandomTransform(object):
    def __init__(self, gs_am, scale_factor, shift_scale, shuffle_rate,
                 max_ratio, n_transform) -> None:
        super().__init__()
        self.gs = ChannelWiseGuassian(gs_am)
        self.sampling = Sampling(scale_factor)
        self.shifting = Shifting(shift_scale)
        self.shuffling = ChannelShuffle(shuffle_rate)
        self.amchange = ChannelAMChange(max_ratio)
        self.transforms = [
            self.gs, self.sampling, self.shifting, self.shuffling
        ]
        self.n_transform = n_transform

    def __call__(self, input):
        transform_idx = np.arange(len(self.transforms))
        transform_idx = np.random.permutation(transform_idx)
        for idx in transform_idx[:self.n_transform]:
            input = self.transforms[idx](input)

        return input


def get_transform(transform_name='random',
                  gs_am=0.5,
                  scale_factor=0.5,
                  shift_scale=0.5,
                  shuffle_rate=0.2,
                  max_ratio=1.5,
                  n_transform=2):
    transform = None
    if transform_name == 'guassian':
        transform = ChannelWiseGuassian(max_am=gs_am)
    elif transform_name == 'sampling':
        transform = Sampling(scale_factor=scale_factor)
    elif transform_name == 'shifting':
        transform = Shifting(shift_scale=shift_scale)
    elif transform_name == 'shuffling':
        transform = ChannelShuffle(shuffle_rate=shuffle_rate)
    elif transform_name == 'amchange':
        transform = ChannelAMChange(max_ratio=max_ratio)
    elif transform_name == 'random':
        transform = RandomTransform(gs_am=gs_am,
                                    scale_factor=scale_factor,
                                    shift_scale=shift_scale,
                                    shuffle_rate=shuffle_rate,
                                    max_ratio=max_ratio,
                                    n_transform=n_transform)
    else:
        raise 'No such transform'

    return transform
