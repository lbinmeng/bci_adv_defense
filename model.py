import copy
import torch
import torch.nn as nn
from typing import Optional


def MaxNormDefaultConstraint(model: nn.Module):
    for n, p in model.named_parameters():
        if n == 'block1.3.weight':
            p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)
        elif n == 'classifier_block.0.weight':
            p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)


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


class Noise(nn.Module):
    ''' noise layer '''
    def __init__(self, std):
        super(Noise, self).__init__()
        self.std = std

    def forward(self, x):
        return x + torch.Tensor(x.size()).normal_(mean=0, std=self.std).to(
            x.device)


class SAP(nn.Module):
    def __init__(self, frac):
        super(SAP, self).__init__()
        self.frac = frac

    def forward(self, x):
        if self.frac is not None:
            shape = x.shape
            prob = x.clone().reshape(shape[0], -1)

            select_num = int(self.frac * prob.shape[1])

            prob = torch.abs(prob)
            prob = prob / torch.sum(prob, dim=1, keepdim=True)
            idx = torch.multinomial(prob, select_num)

            x = x.reshape(shape[0], -1)
            # pruned
            scale_factor = torch.zeros_like(x).to(x.device)
            selected = torch.gather(prob, dim=1, index=idx)
            selected = 1.0 / (1.0 - torch.pow(1 - selected, select_num) + 1e-8)
            scale_factor = torch.scatter(scale_factor, dim=1, index=idx, src=selected)
            x = scale_factor * x
            x = x.reshape(shape)
        return x


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
                 norm_rate: Optional[float] = 0.25,
                 noise_std: Optional[float] = None,
                 SAP_frac: Optional[float] = None):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.noise_std = noise_std
        self.SAP_frac = SAP_frac
        if noise_std is not None:
            self.noise_layer1 = Noise(self.noise_std * 2)
            self.noise_layer2 = Noise(self.noise_std)

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            SAP(frac=self.SAP_frac),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            SAP(frac=self.SAP_frac),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std is None:
            output = self.block1(x)
        else:
            output = self.noise_layer1(x)
            output = self.block1(output)
            output = self.noise_layer2(output)
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
                 dropoutRate: Optional[float] = 0.5,
                 noise_std: Optional[float] = None,
                 SAP_frac: Optional[float] = None):
        super(DeepConvNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.noise_std = noise_std
        self.SAP_frac = SAP_frac
        if noise_std is not None:
            self.noise_layer1 = Noise(self.noise_std * 2)
            self.noise_layer2 = Noise(self.noise_std)
            self.noise_layer3 = Noise(self.noise_std)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(num_features=25), nn.ELU(), SAP(frac=self.SAP_frac),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=50), nn.ELU(), SAP(frac=self.SAP_frac),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=100), nn.ELU(),
            SAP(frac=self.SAP_frac),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=100 *
                      CalculateOutSize([self.block1, self.block2, self.block3],
                                       self.Chans, self.Samples),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std is None:
            output = self.block1(x)
            output = self.block2(output)
        else:
            output = self.noise_layer1(x)
            output = self.block1(output)
            output = self.noise_layer2(output)
            output = self.block2(output)
            output = self.noise_layer3(output)
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
                 dropoutRate: Optional[float] = 0.5,
                 noise_std: Optional[float] = None,
                 SAP_frac: Optional[float] = None):
        super(ShallowConvNet, self).__init__()
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.noise_std = noise_std
        self.SAP_frac = SAP_frac
        if noise_std is not None: self.noise_layer = Noise(self.noise_std * 2)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=40,
                      out_channels=40,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=40), Activation('square'),
            SAP(frac=self.SAP_frac),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            Activation('log'), SAP(frac=self.SAP_frac),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            nn.Linear(
                in_features=40 *
                CalculateOutSize([self.block1], self.Chans, self.Samples),
                out_features=self.n_classes,
                bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std is None:
            output = self.block1(x)
        else:
            output = self.noise_layer(x)
            output = self.block1(output)
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


class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        decay = min(self.alpha, 1 - 1 / (self.step + 25))
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] +
                                    (1 - decay) * state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] +
                                        (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }
