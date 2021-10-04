import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# library of loss functions that could potentially tackle class imbalance

# Asymmetric/ Focal loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        # print(loss)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            # print(pt)
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.0, alpha = 0.25, reduction='none', eps = 1e-8):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target, alphas=None):
        cross_entropy_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        # print(cross_entropy_loss)
        # cross_entropy_loss = nn.BCELoss(target, input)
        p_t = ((target * torch.sigmoid(input)) +
               ((1 - target) * (1 - torch.sigmoid(input))))
        # print(p_t)
        modulating_factor = 1.0
        if self.gamma:
            modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        alpha_weight_factor = 1.0
        if self.alpha is not None:
            alpha_weight_factor = (target * self.alpha +
                                   (1 - target) * (1 - self.alpha))
            # print(alpha_weight_factor)
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    cross_entropy_loss)
        return focal_cross_entropy_loss.sum()

class CBLoss(nn.Module):
    def __init__(self, gamma = 2.0, reduction='none', eps = 1e-8):
        super(CBLoss, self).__init__()

        self.gamma = gamma

    def forward(self, input, target, weights=None):
        # compute focal loss
        cross_entropy_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        p_t = ((target * torch.sigmoid(input)) +
               ((1 - target) * (1 - torch.sigmoid(input))))

        modulating_factor = 1.0
        if self.gamma:
            modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        alpha_weight_factor = 1.0
        if weights is not None:
            alpha_weight_factor = (target * weights +
                                   (1 - target) * (1 - weights))

        CB_focal_loss = (modulating_factor * alpha_weight_factor *
                                    cross_entropy_loss)
        return CB_focal_loss.sum()
