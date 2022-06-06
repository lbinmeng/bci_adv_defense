from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class NoisyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(NoisyBatchNorm2d, self).__init__(num_features, eps, momentum,
                                               affine, track_running_stats)
        self.neuron_mask = Parameter(torch.Tensor(num_features))
        self.neuron_noise = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask)
        init.zeros_(self.neuron_noise)
        init.zeros_(self.neuron_noise_bias)
        self.is_perturbed = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.neuron_noise, a=-eps, b=eps)
            init.uniform_(self.neuron_noise_bias, a=-eps, b=eps)
        else:
            init.zeros_(self.neuron_noise)
            init.zeros_(self.neuron_noise_bias)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is
                                                           None)
        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean,
                                                       torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var,
                                                      torch.Tensor)

        if self.is_perturbed:
            coeff_weight = self.neuron_mask + self.neuron_noise
            coeff_bias = 1.0 + self.neuron_noise_bias
        else:
            coeff_weight = self.neuron_mask
            coeff_bias = 1.0
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats else None,
            self.running_var
            if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight,
            self.bias * coeff_bias,
            bn_training,
            exponential_average_factor,
            self.eps)


class MaskConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None) -> None:
        super(MaskConv2d, self).__init__(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias,
                                         padding_mode=padding_mode,
                                         device=device,
                                         dtype=dtype)
        self.neuron_mask = Parameter(torch.Tensor(self.weight.shape))
        init.ones_(self.neuron_mask)

    def forward(self, input):
        return F.conv2d(input, self.weight * self.neuron_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class MaskLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype=None) -> None:
        super(MaskLinear, self).__init__(in_features,
                                         out_features,
                                         bias=bias,
                                         device=device,
                                         dtype=dtype)
        self.neuron_mask = Parameter(torch.Tensor(self.weight.shape))
        init.ones_(self.neuron_mask)

    def forward(self, input):
        return F.linear(input, self.weight * self.neuron_mask, self.bias)


def CalculateOutSize(blocks, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    x = torch.rand(1, 1, channels, samples)
    for block in blocks:
        block.eval()
        x = block(x)
    shape = x.shape[-2] * x.shape[-1]
    return shape


class EEGNet(nn.Module):
    """
    :param
    """
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: Optional[float] = 0.5,
                 norm_rate: Optional[float] = 0.25):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            MaskConv2d(in_channels=1,
                       out_channels=self.F1,
                       kernel_size=(1, self.kernLenght),
                       stride=1,
                       bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            MaskConv2d(in_channels=self.F1,
                       out_channels=self.F1 * self.D,
                       kernel_size=(self.Chans, 1),
                       groups=self.F1,
                       bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            MaskConv2d(in_channels=self.F1 * self.D,
                       out_channels=self.F1 * self.D,
                       kernel_size=(1, 16),
                       stride=1,
                       groups=self.F1 * self.D,
                       bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            MaskConv2d(in_channels=self.F1 * self.D,
                       out_channels=self.F2,
                       kernel_size=(1, 1),
                       stride=1,
                       bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            MaskLinear(in_features=self.F2 * (self.Samples // (4 * 8)),
                       out_features=self.n_classes,
                       bias=True))
        # nn.ELU(),
        # nn.Linear(in_features=50, out_features=self.n_classes, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)


class DeepConvNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5):
        super(DeepConvNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            MaskConv2d(in_channels=1, out_channels=25, kernel_size=(1, 5)),
            MaskConv2d(in_channels=25, out_channels=25,
                       kernel_size=(Chans, 1)),
            nn.BatchNorm2d(num_features=25), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            MaskConv2d(in_channels=25, out_channels=50, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=50), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            MaskConv2d(in_channels=50, out_channels=100, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=100), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            MaskLinear(
                in_features=100 *
                CalculateOutSize([self.block1, self.block2, self.block3],
                                 self.Chans, self.Samples),
                out_features=self.n_classes,
                bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)


class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


class ShallowConvNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5):
        super(ShallowConvNet, self).__init__()
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            MaskConv2d(in_channels=1, out_channels=40, kernel_size=(1, 13)),
            MaskConv2d(in_channels=40,
                       out_channels=40,
                       kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=40), Activation('square'),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            Activation('log'), nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            MaskLinear(
                in_features=40 *
                CalculateOutSize([self.block1], self.Chans, self.Samples),
                out_features=self.n_classes,
                bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.5)