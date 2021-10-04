# list of custom modules that may be of use when building the Generator and
# Discriminator Networks.

# These are separate to the custom modules/layers already provided in the
# FastGAN code unless needed

# code from various sources

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch import einsum
from einops import rearrange, repeat
import random
from kornia.filters import filter2d
from math import floor, log2

from torchvision import transforms

# list of custom modules that may be of use when building the Generator and
# Discriminator Networks.

# These are separate to the custom modules/layers already provided in the
# FastGAN code unless needed

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

############################## Normalization ###################################
class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))



############################## Bilinear Filtering ##############################

# https://github.com/lucidrains/lightweight-gan
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1,2,1])
        self.register_buffer('f',f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None,:,None]
        return filter2d(x, f, normalized = True)

############################## Channel Attention ###############################

# Global Context Attention on channels for SLE
# https://github.com/lucidrains/lightweight-gan
class GlobalContextSL(nn.Module):
    def __init__(self,*, ch_in, ch_out):
        super().__init__()
        self.to_k = nn.Conv2d(ch_in, 1, 1)
        ch_intermediate = max(3, ch_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ch_intermediate, ch_out, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim = -1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)

# original global context - not skipping layers
# https://github.com/xvjiarui/GCNet
def kaiming_init(module,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)

class GlobalContext(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

############################# Spatial Attention ################################
# Non-local Attention
# Adaptation from SA-GAN used in BigGAN
# https://github.com/ajbrock/BigGAN-PyTorch
class NonLocalAttention(nn.Module):
    def __init__(self, ch_in):
        super(NonLocalAttention, self).__init__()
        # Channel lr_multiplier
        self.ch_in = ch_in
        self.ch_attn = ch_in // 8

        self.theta = conv2d(self.ch_in, self.ch_attn, kernel_size=1, padding=0, bias=False)
        self.phi = conv2d(self.ch_in, self.ch_attn, kernel_size=1, padding=0, bias=False)
        self.g = conv2d(self.ch_in, self.ch_in // 2, kernel_size=1, padding=0, bias=False)
        self.o = conv2d(self.ch_in // 2, self.ch_in, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma =  nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        # Perform reshapes
        theta = theta.view(-1, self.ch_in // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch_in // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch_in // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch_in // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

# Efficient Attention with linear complexity
# Adapted to have learnable scaling parameter
# https://github.com/cmsflash/efficient-attention
class LinearAttention(nn.Module):
    def __init__(self, ch_in, ch_key, ch_val, head_count):
        super().__init__()
        self.ch_in = ch_in
        self.ch_key = ch_key
        self.head_count = head_count
        self.ch_val = ch_val

        # use spectral normalized conv2d
        self.keys = conv2d(ch_in, ch_key, 1)
        self.queries = conv2d(ch_in, ch_key, 1)
        self.values = conv2d(ch_in, ch_val, 1)
        self.reprojection = conv2d(ch_val, ch_in, 1)

        # learnable gain parameters
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.ch_key, h * w))
        queries = self.queries(input_).reshape(n, self.ch_key, h * w)
        values = self.values(input_).reshape((n, self.ch_val, h * w))
        head_key_channels = self.ch_key // self.head_count
        head_value_channels = self.ch_val // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        # attention map
        reprojected_value = self.reprojection(aggregated_values)

        return self.gamma * reprojected_value + input_

# Separate implementation of efficient Attention
# https://github.com/lucidrains/lightweight-gan
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LinearAttentionAlt(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out) + fmap
