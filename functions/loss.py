import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def flatten(tensor):
    c = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order).contiguous()
    return transposed.view(c, -1)


class SoftDiceLoss(nn.Module):

    def __init__(self, ignore_index=None, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, output, target):
        output = torch.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target).float()

        intersect = (output * target).sum(-1)
        denominator = output.sum(-1) + target.sum(-1)

        if self.ignore_index is not None:
            intersect = torch.cat(
                (intersect[0:self.ignore_index],
                intersect[self.ignore_index + 1:])
            )
            denominator = torch.cat(
                (denominator[0:self.ignore_index],
                denominator[self.ignore_index + 1:])
            )

        intersect = intersect.sum()
        denominator = denominator.sum()
        dice = 2.0 * intersect / denominator.clamp(min=self.smooth)
        return 1.0 - dice


class GeneralizedDiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, output, target):
        output = torch.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target).float()
        target_sum = target.sum(-1)

        class_weights = 1.0 / (target_sum * target_sum).clamp(min=self.smooth)
        class_weights.requires_grad_(False)

        intersect = ((output * target).sum(-1) * class_weights).sum()
        denominator = ((output + target).sum(-1) * class_weights).sum()

        return 1.0 - 2.0 * intersect / denominator.clamp(min=self.smooth)


class FocalLoss(nn.Module):
    epsilon = 1e-6

    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target):
        # output: (B, C, H, W)
        # target: (B, C, H, W)
        p = torch.softmax(output, dim=1)
        p = torch.clamp(p, min=self.epsilon, max=1-self.epsilon)
        log_p = torch.log_softmax(output, dim=1)

        loss_sce = - target * log_p
        loss_focal = torch.sum(loss_sce * (1.0 - p) ** self.gamma, dim=1)

        return loss_focal.mean()


class OneHotEncoder(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.ones = torch.sparse.torch.eye(n_classes).cuda()

    def forward(self, t):
        n_dim = t.dim()
        output_size = t.size() + torch.Size([self.n_classes])

        t = t.data.long().contiguous().view(-1).cuda()
        out = Variable(self.ones.index_select(0, t)).view(output_size)
        out = out.permute(0, -1, *range(1, n_dim)).contiguous().float()

        return out
