import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer, surrogate
from spikingjelly.clock_driven.neuron import BaseNode, LIFNode
from torchvision import transforms
import math

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1),
                                        LLC_LIFNode(init_tau=2.0,
                                                    surrogate_function=surrogate.ATan(learnable=False),
                                                    detach_reset=True))

        self.key_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1),
                                      LLC_LIFNode(init_tau=2.0,
                                                  surrogate_function=surrogate.ATan(learnable=False),
                                                  detach_reset=True))
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
                                        LLC_LIFNode(init_tau=2.0,
                                                    surrogate_function=surrogate.ATan(learnable=False),
                                                    detach_reset=True))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        # energy = self.dropout(energy)
        attention = self.softmax(energy)  # BX (N) X (N)
        attention = self.dropout(attention)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out


class llcBatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=3.2, affine=True, track_running_stats=True,
                 init_tau=2.0):
        super(llcBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        init_w = - math.log(init_tau - 1.0)
        # init_w2 = - math.log(init_tau-1.0)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))
        # self.alpha = nn.Parameter(torch.tensor(init_w2, dtype=torch.float))

    def forward(self, input):
        exponential_average_factor = 0.0
        if self.training:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            mean = input.mean([0, 2, 3])
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (
                        1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) + (
                        1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = self.alpha * self.w.sigmoid() * (input - mean[None, :, None, None]) / (
            torch.sqrt(var[None, :, None, None] + self.eps))

        # print("self.alpha * self.w.sigmoid():", self.alpha * self.w.sigmoid())
        input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return input


class LLC_LIFNode(BaseNode):
    def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=True,
                 surrogate_function=surrogate.ATan(), monitor_state=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        init_w = - math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            self.v += (dv - self.v) * self.w.sigmoid()
        else:
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        # print(self.w.sigmoid())
        return self.spiking()

    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):

        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'


def create_conv_sequential(in_channels, out_channels, number_layer, init_tau, use_LLC_LIF, use_max_pool,
                           alpha_learnable, detach_reset):

    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                    detach_reset=detach_reset) if use_LLC_LIF else LIFNode(tau=init_tau,
                                                                           surrogate_function=surrogate.ATan(
                                                                               learnable=alpha_learnable),
                                                                           detach_reset=detach_reset),
        nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
    ]

    for i in range(number_layer - 1):
        conv.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(tau=init_tau,
                                                                               surrogate_function=surrogate.ATan(
                                                                                   learnable=alpha_learnable),
                                                                               detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        ])
    return nn.Sequential(*conv)


def create_2fc(channels, h, w, dpp, class_num, init_tau, use_LLC_LIF, alpha_learnable, detach_reset):
    return nn.Sequential(
        nn.Flatten(),
        layer.Dropout(dpp),
        nn.Linear(channels * h * w, channels * h * w // 4, bias=False),
        LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                    detach_reset=detach_reset) if use_LLC_LIF else LIFNode(tau=init_tau,
                                                                           surrogate_function=surrogate.ATan(
                                                                               learnable=alpha_learnable),
                                                                           detach_reset=detach_reset),
        layer.Dropout(dpp, dropout_spikes=True),
        nn.Linear(channels * h * w // 4, class_num * 10, bias=False),
        LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                    detach_reset=detach_reset) if use_LLC_LIF else LIFNode(tau=init_tau,
                                                                           surrogate_function=surrogate.ATan(
                                                                               learnable=alpha_learnable),
                                                                           detach_reset=detach_reset),
    )


class StaticNetBase(nn.Module):
    def __init__(self, T, init_tau, use_LLC_LIF, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_LLC_LIF = use_LLC_LIF
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset
        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.static_conv = None
        self.conv = None
        self.conv_local_fc2 = None
        self.basic_conv = None
        self.conv_local_fc1 = None
        self.fc = None
        self.fc_s1 = None
        self.fc_s2 = None
        self.boost = nn.AvgPool1d(10, 10)
        self.local_fc1 = None
        self.local_fc2 = None

        self.fc_add = None
        # self.attn = Self_Attn()

    def forward(self, x):
        x = self.static_conv(x)
        out_spikes_counter_global = self.boost(self.fc(self.conv0(x)).unsqueeze(1)).squeeze(1)
        out_spikes_counter_local1 = self.boost((self.conv_local_fc1(x)).unsqueeze(1)).squeeze(1)
        out_spikes_counter_local2 = self.boost(self.conv_local_fc2(self.conv0(x)).unsqueeze(1)).squeeze(1)

        for t in range(1, self.T):
            out_spikes_counter_global += self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)
            out_spikes_counter_local1 += self.boost(self.conv_local_fc1(x).unsqueeze(1)).squeeze(1)
            out_spikes_counter_local2 += self.boost(self.conv_local_fc2(self.conv(x)).unsqueeze(1)).squeeze(1)

        return out_spikes_counter_local1, out_spikes_counter_local2, out_spikes_counter_global


class MNISTNet(StaticNetBase):
    def __init__(self, T, init_tau, use_LLC_LIF, use_max_pool, alpha_learnable, detach_reset):
        super().__init__(T, init_tau=init_tau, use_LLC_LIF=use_LLC_LIF, use_max_pool=use_max_pool,
                         alpha_learnable=alpha_learnable, detach_reset=detach_reset)

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(128),
        )

        self.conv0 = nn.Sequential(
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            Self_Attn(128),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),

            llcBatchNorm(128),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        )

        self.conv = nn.Sequential(
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            # Self_Attn(128),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),

            llcBatchNorm(128),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        )

        self.conv_local_fc1 = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(128 * 28 * 28, 100, bias=False),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

        )

        self.conv_local_fc2 = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(128 * 7 * 7, 100, bias=False),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

        )

        # self.fc = nn.Sequential(self.fc_s1, self.fc_s2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            layer.Dropout(0.5, dropout_spikes=True),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset)
        )
        # self.fc_add = nn.Sequential(self.fc_s1, self.fclocal)

        # self.globalnet = nn.Sequential(self.globalconv, self.fcglobal)


class FashionMNISTNet(MNISTNet):
    pass


class Cifar10Net(StaticNetBase):
    def __init__(self, T, init_tau, use_LLC_LIF, use_max_pool, alpha_learnable, detach_reset):
        super().__init__(T, init_tau=init_tau, use_LLC_LIF=use_LLC_LIF, use_max_pool=use_max_pool,
                         alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.static_conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256)
        )
        self.conv0 = nn.Sequential(
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            Self_Attn(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)

        )
        self.conv = nn.Sequential(
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            llcBatchNorm(256),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)

        )

        self.conv_local_fc1 = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(256 * 32 * 32, 100, bias=False),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

        )

        self.conv_local_fc2 = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(256 * 8 * 8, 100, bias=False),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(256 * 8 * 8, 128 * 4 * 4, bias=False),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            LLC_LIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable),
                        detach_reset=detach_reset) if use_LLC_LIF else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset)
        )


def get_transforms(dataset_name):
    transform_train = None
    transform_test = None
    if dataset_name == 'MNIST':
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])
    elif dataset_name == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.2860, 0.3530),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.2860, 0.3530),
        ])
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform_train, transform_test



