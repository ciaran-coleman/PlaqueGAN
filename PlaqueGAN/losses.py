# adversarial loss variants that might be useful for PlaqueGAN
# regular hinge loss, relativistic average hinge loss, EMA anchored loss

# Adapted from the official implementation of the paper:
# "Regularizing Generative Adversarial Networks under Limited Data" by Tseng et al.
# https://github.com/google/lecam-gan/
import torch
import torch.nn.functional as F

# Simple wrapper that applies EMA to losses.
class ema_losses(object):
    def __init__(self, init=1000., decay=0.9, start_iter=0):
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay
        self.start_iter = start_iter

    def update(self, cur, mode, iter):
        if iter < self.start_iter:
            decay = 0.0
        else:
            decay = self.decay
        if mode == 'G_loss':
          self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == 'D_loss_real':
          self.D_loss_real = self.D_loss_real*decay + cur*(1 - decay)
        elif mode == 'D_loss_fake':
          self.D_loss_fake = self.D_loss_fake*decay + cur*(1 - decay)
        elif mode == 'D_real':
          self.D_real = self.D_real*decay + cur*(1 - decay)
        elif mode == 'D_fake':
          self.D_fake = self.D_fake*decay + cur*(1 - decay)

# LeCam Regularization loss
def lecam_reg(dis_real, dis_fake, ema):
    reg = torch.mean(F.relu(dis_real - ema.D_fake).pow(2)) + \
        torch.mean(F.relu(ema.D_real - dis_fake).pow(2))
    return reg

# ------ non-saturated ------ #
def loss_dcgan_dis(dis_fake, dis_real, ema=None, it=None):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2

def loss_dcgan_gen(dis_fake, dis_real=None):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss

# ------ lsgan ------ #
def loss_ls_dis(dis_fake, dis_real, ema=None, it=None):
  loss_real = torch.mean((dis_real + 1).pow(2))
  loss_fake = torch.mean((dis_fake - 1).pow(2))
  return loss_real, loss_fake

def loss_ls_gen(dis_fake, dis_real=None):
  return torch.mean(dis_fake.pow(2))

# ------ rahinge ------ #
def loss_rahinge_dis(dis_fake, dis_real, ema=None, it=None, label_smooth=0.):
    r_f_diff = dis_real - torch.mean(dis_fake)
    f_r_diff = dis_fake - torch.mean(dis_real)

    loss_real = torch.mean(F.relu(torch.rand_like(dis_real)*label_smooth + (1 - label_smooth) - r_f_diff))/2
    loss_fake = torch.mean(F.relu(torch.rand_like(dis_real)*label_smooth + (1 - label_smooth) + f_r_diff))/2

    return loss_real, loss_fake

def loss_rahinge_gen(dis_fake, dis_real, label_smooth=0.):
    r_f_diff = dis_real - torch.mean(dis_fake)
    f_r_diff = dis_fake - torch.mean(dis_real)

    loss_real = torch.mean(F.relu(torch.rand_like(dis_real)*label_smooth + (1 - label_smooth) + r_f_diff))
    loss_fake = torch.mean(F.relu(torch.rand_like(dis_fake)*label_smooth + (1 - label_smooth) - f_r_diff))

    return (loss_real + loss_fake)/2

# ------ hinge ------ #
def loss_hinge_dis(dis_fake, dis_real, ema=None, it=None, label_smooth=0.):
    if ema is not None:
        # track the prediction
        ema.update(torch.mean(dis_fake).item(), 'D_fake', it)
        ema.update(torch.mean(dis_real).item(), 'D_real', it)

    loss_real = F.relu(torch.rand_like(dis_real)*label_smooth + (1 - label_smooth) - dis_real).mean()
    loss_fake = F.relu(torch.rand_like(dis_fake)*label_smooth + (1 - label_smooth) + dis_fake).mean()

    return loss_real, loss_fake

def loss_hinge_gen(dis_fake, dis_real=None, label_smooth=0.):
    loss = -dis_fake.mean()
    return loss
