# code for building PlaqueGAN generator and discriminator.
# Adapted from code of official FastGAN implementation:
# https://github.com/odegeasslbc/FastGAN-pytorch
# with additional adaptations from:
# https://github.com/lucidrains/lightweight-gan

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torch import einsum
import random

from math import log2
from torchvision import transforms

seq = nn.Sequential

# spatial self-attention
from custom_modules import NonLocalAttention, LinearAttention, LinearAttentionAlt
# channel Attention
from custom_modules import GlobalContext, GlobalContextSL #SL is for applying between layers with different num_channels

from custom_modules import Blur, ChanNorm, PreNorm

def exists(val):
    return val is not None

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

class Mish(nn.Module):
    def forward(self, feat):
        return feat * torch.tanh(F.softplus(feat))

# adapted to forward the residual path (features of lower resolution)
class SLE(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4),
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, x):
        return self.main(x)

class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        batchNorm2d(channel*2), nn.GLU(dim=1) )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)

# combine both UpBlock and UpBlockComp from FastGAN implementation
def UpBlock(in_planes, out_planes, conv_layers = 1, anti_alias = False,
            noise_inj = False):

    block_1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        Blur() if anti_alias else nn.Identity(),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection() if noise_inj else nn.Identity(),
        batchNorm2d(out_planes*2), nn.GLU(dim=1))

    if conv_layers>1:
        block_2 = nn.Sequential(
            conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
            NoiseInjection() if noise_inj else nn.Identity(),
            batchNorm2d(out_planes*2), nn.GLU(dim=1))

        return nn.Sequential(*(list(block_1)+list(block_2)))
    else:
        return nn.Sequential(*list(block_1))

class Generator(nn.Module):
    def __init__(self,
                 nz                 = 256,      # latent dimension
                 nc                 = 3,        # number of channels
                 im_size            = 256,
                 activation         = 'lrelu',   # alternatives 'swish' or 'mish'
                 chan_attn          = 'SLE',
                 sle_map            = [(4,64),(8,128),(16,256)],
                 skip_conn          = [0,1,2],
                 attn_layers        = [64],
                 spatial_attn       = None,
                 conv_layers        = 1,        # number of conv layers per block
                 alternate_layers   = True,     # if True, alternate between 1 and 2 conv layers
                 anti_alias         = False,
                 noise_inj          = True,
                 multi_res_out      = False,    # if True, outputs smaller image size as well
                 small_im_size      = 128,       # downsampled image size
                 use_tanh           = True

    ):
        super(Generator, self).__init__()

        self.im_size = im_size
        self.spatial_attn = spatial_attn
        self.chan_attn = chan_attn
        self.conv_layers = conv_layers
        self.alternate_layers = alternate_layers
        self.anti_alias = anti_alias
        self.noise_inj = noise_inj
        self.multi_res_out = multi_res_out
        self.small_im_size = small_im_size
        self.use_tanh = use_tanh

        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif activation == 'swish':
            self.act = Swish()

        resolution = int(log2(self.im_size))
        num_layers = resolution - 2 # because we start off at resolution = 4

        # resolution layers
        self.res_layers = range(2, num_layers + 2)
        in_out_channels = list(map(lambda x: (2**(12-x),2**(11-x)), self.res_layers))
        self.res_to_channels_map = dict(zip(self.res_layers, in_out_channels))

        self.size_to_channels_map = dict(list(map(lambda x:(2**x, 2**(12-x)), range(2, num_layers+3))))
        # skip-layer Mapping
        self.sle_map = [(log2(low_res), log2(hi_res)) for low_res, hi_res in sle_map]

        self.sle_map = [x for i,x in enumerate(self.sle_map) if i in skip_conn]
        self.sle_map = list(map(lambda x: (int(x[0]), int(x[1])), self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.init = InitLayer(nz, channel = self.size_to_channels_map[4])
        self.layers = nn.ModuleList([])

        for (res, (ch_in, ch_out)) in zip(self.res_layers, in_out_channels):
            current_im_size = 2 ** res

            # set skip-layer excitation and spatial attention off by default
            sle = None
            attn = None

            # create attention block if requested
            if current_im_size in attn_layers:
                if self.spatial_attn == 'linear':
                    attn = LinearAttention(ch_in = ch_in, ch_key=64, ch_val=64, head_count = 1)
                elif self.spatial_attn == 'linear_alt':
                    attn = LinearAttentionAlt(ch_in)
                elif self.spatial_attn == 'nonlocal':
                    attn = NonLocalAttention(ch_in = ch_in)
                elif self.spatial_attn == 'global':
                    attn = GlobalContext(inplanes = ch_in, ratio = 1/8, fusion_types = 'channel_add')

            # create skip-layer excitation block if requested
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_ch_out = self.res_to_channels_map[residual_layer-1][-1]

                if self.chan_attn == 'SLE':
                    sle = SLE(ch_in = ch_out, ch_out = sle_ch_out)
                elif self.chan_attn == 'GC':
                    sle = GlobalContextSL(ch_in = ch_out, ch_out = sle_ch_out)

            if self.alternate_layers:
                layer = nn.ModuleList([
                    UpBlock(ch_in, ch_out, conv_layers = (1 if (res%2==0) else 2), anti_alias = self.anti_alias,
                                noise_inj = self.noise_inj),
                    sle,
                    attn
                    ])
            else:
                layer = nn.ModuleList([
                    UpBlock(ch_in, ch_out, conv_layers = self.conv_layers, anti_alias = self.anti_alias,
                                noise_inj = self.noise_inj),
                    sle,
                    attn
                ])
            self.layers.append(layer)
            # output layer to RGB
            self.out_conv = conv2d(self.size_to_channels_map[self.im_size], nc, 3, 1, 1, bias=True)
            if self.multi_res_out:
                self.small_out_conv = conv2d(self.size_to_channels_map[self.small_im_size], nc, 1, 1, 0, bias=True)

    def forward(self, input):
        x = self.init(input)
        residuals = dict()

        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):
            if exists(attn):
                x = attn(x)

            if 2**res == self.small_im_size:
                small_feats = x

            x = up(x)

            # this method might require more GPU memory than before as it is storing the residuals
            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res] # combining skip-layer paths

        if not self.multi_res_out:
            return [torch.tanh(self.out_conv(x)) if self.use_tanh else self.out_conv(x)] # return as a list to simplify
        else:
            return [torch.tanh(self.out_conv(x)), torch.tanh(self.small_out_conv(small_feats))] if self.use_tanh else [self.out_conv(x), self.small_out_conv(small_feats)]

# Discriminator
class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, activation = nn.LeakyReLU(0.2), anti_alias = False):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            Blur() if anti_alias else nn.Identity(),
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), activation,
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes, activation= nn.LeakyReLU(0.2), anti_alias = False):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            Blur() if anti_alias else nn.Identity(),
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), activation,
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), activation
            )

        self.direct = nn.Sequential(
            Blur() if anti_alias else nn.Identity(), # blur might not be needed here
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), activation)

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2

# should probably include attention in this block if its going to be used
class DownFromSmall(nn.Module):
    def __init__(self, ch_in, im_size, size_to_ch, activation = nn.LeakyReLU(0.2), anti_alias = False,
                use_attn = False, spatial_attn = 'linear'):
        super(DownFromSmall, self).__init__()
        self.layers = nn.ModuleList([])
        # ch_final = ch_out
        num_downsamples = int(log2(im_size//8)) - 1

        self.first_conv = conv2d(ch_in, size_to_ch[im_size], 4, 2, 1)

        self.first_attn = None
        self.final_attn = None

        if use_attn:
            if spatial_attn == 'linear':
                self.first_attn = LinearAttention(ch_in = size_to_ch[im_size], ch_key=64, ch_val=64, head_count = 1)
            elif spatial_attn == 'linear_alt':
                self.first_attn = LinearAttentionAlt(size_to_ch[im_size])
            elif spatial_attn == 'nonlocal':
                self.first_attn = NonLocalAttention(ch_in = size_to_ch[im_size])
            elif spatial_attn == 'global':
                self.first_attn = GlobalContext(inplanes = size_to_ch[im_size], ratio = 1/8, fusion_types = 'channel_add')


        for ind in range(num_downsamples):

            self.layers.append(DownBlock(size_to_ch[im_size], size_to_ch[im_size//2], activation = activation, anti_alias = anti_alias))
            im_size = im_size//2

        if use_attn:
            if spatial_attn == 'linear':
                self.final_attn = LinearAttention(ch_in = size_to_ch[im_size], ch_key=64, ch_val=64, head_count = 1)
            elif spatial_attn == 'linear_alt':
                self.final_attn = LinearAttentionAlt(size_to_ch[im_size])
            elif spatial_attn == 'nonlocal':
                self.final_attn = NonLocalAttention(ch_in = size_to_ch[im_size])
            elif spatial_attn == 'global':
                self.final_attn = GlobalContext(inplanes = size_to_ch[im_size], ratio = 1/8, fusion_types = 'channel_add')

    def forward(self, x):
        x = self.first_conv(x)
        if exists(self.first_attn):
            x = self.first_attn(x)
        for layer in self.layers:
            x = layer(x)
        if exists(self.final_attn):
            x = self.final_attn(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,
                 nc                 = 3,        # number of channels
                 im_size            = 256,
                 activation         = 'lrelu',   # alternatives 'swish' or 'mish'
                 chan_attn          = 'SLE',
                 sle_map            = [(256,32),(128,16),(64,8)],
                 skip_conn          = [],  # skip-layer connections to use. Original paper has none in discriminator
                 attn_layers        = [],
                 spatial_attn       = None,
                 attn_dis2          = False,
                 anti_alias         = False,
                 out_logits_size    = 5,        # alternative is 1
                 multi_logits_out   = True,      # whether to discrimate with a downsampled image
                 small_im_size      = 128,       # downsampled image size
                 d_num_rec          = 2,
                 use_tanh           = True,
                 minibatch_d        = False,    # use minibatch discrimination module
                 ):

        super(Discriminator, self).__init__()

        self.im_size = im_size
        self.spatial_attn = spatial_attn
        self.chan_attn = chan_attn
        self.anti_alias = anti_alias
        self.multi_logits_out = multi_logits_out
        self.attn_layers = attn_layers
        if self.multi_logits_out:
            self.small_im_size = small_im_size
        self.d_num_rec = d_num_rec
        self.use_tanh = use_tanh
        self.minibatch_d = minibatch_d
        self.attn_dis2 = attn_dis2
        print(self.attn_dis2)
        if activation == 'lrelu':
            act = nn.LeakyReLU(0.2)
        elif activation == 'swish':
            act = Swish()

        resolution = int(log2(self.im_size))
        num_residual_layers = 8 - 3
        self.non_residual_resolutions = range(min(8, resolution), 2, -1)
        channels = list(map(lambda x: (x,  2 ** (12 - x)), self.non_residual_resolutions))
        in_out_channels = list(map(lambda x: (2**(12-x),2**(13-x)), self.non_residual_resolutions[:-1]))
        self.res_to_channels_map = dict(zip(self.non_residual_resolutions, in_out_channels))
        self.size_to_channels_map = dict(list(map(lambda x: (2**x, 2**(12-x)),self.non_residual_resolutions)))

        # skip-layer Mapping
        self.sle_map = [(log2(hi_res), log2(low_res)) for hi_res, low_res in sle_map]
        self.sle_map = [x for i,x in enumerate(self.sle_map) if i in skip_conn]
        self.sle_map = list(map(lambda x: (int(x[0]), int(x[1])), self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.down_from_big = nn.Sequential(
                                conv2d(nc, in_out_channels[0][0], 3, 1, 1),
                                act)

        self.layers = nn.ModuleList([])

        for (res,(ch_in, ch_out)) in zip(self.non_residual_resolutions[:-1], in_out_channels):
            current_im_size = 2 ** res

            # set skip-layer excitation and spatial attention to off by default
            sle = None
            attn = None

            if current_im_size in attn_layers:
                if self.spatial_attn == 'linear':
                    attn = LinearAttention(ch_in = ch_in, ch_key=64, ch_val=64, head_count = 1)
                elif self.spatial_attn == 'linear_alt':
                    attn = LinearAttentionAlt(ch_in)
                elif self.spatial_attn == 'nonlocal':
                    attn = NonLocalAttention(ch_in = ch_in)
                elif self.spatial_attn == 'global':
                    attn = GlobalContext(inplanes = ch_in, ratio = 1/8, fusion_types = 'channel_add')

            # create skip-layer excitation block if requested
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_ch_out = self.res_to_channels_map[residual_layer+1][-1]

                if self.chan_attn == 'SLE':
                    sle = SLE(ch_in = ch_in, ch_out = sle_ch_out)
                elif self.chan_attn == 'GC':
                    sle = GlobalContextSL(ch_in = ch_in, ch_out = sle_ch_out)

            layer = nn.ModuleList([
                DownBlockComp(ch_in, ch_out, activation = act, anti_alias = self.anti_alias),
                sle,
                attn
            ])
            self.layers.append(layer)

            last_chan = channels[-1][-1]
            if self.minibatch_d:
                last_chan += 1

            if out_logits_size == 5:
                self.to_logits = nn.Sequential(
                    conv2d(last_chan, last_chan, 1, 1, 0, bias=False),
                    batchNorm2d(last_chan), act,
                    conv2d(last_chan, 1, 4, 1, 0)
                )
            elif out_logits_size == 1:
                self.to_logits = nn.Sequential(
                    Blur() if self.anti_alias else nn.Identity(),
                    conv2d(last_chan, last_chan, 3, 2, 1, bias=False),
                    batchNorm2d(last_chan), act,
                    conv2d(last_chan, 1, 4, 1, 0)
                )

        # second feature extractor from a downsampled version of the image
        if self.multi_logits_out:
            last_chan_small = channels[-2][-1]
            if self.minibatch_d:
                last_chan_small += 1

            self.down_from_small = DownFromSmall(nc,
                                                self.small_im_size,
                                                self.size_to_channels_map,
                                                activation = act,
                                                anti_alias = self.anti_alias,
                                                use_attn = self.attn_dis2,
                                                spatial_attn = self.spatial_attn)

            if out_logits_size == 5:
                self.to_logits_small = conv2d(last_chan_small, 1, 4, 1, 0, bias=True)

            elif out_logits_size == 1:
                self.to_logits_small = nn.Sequential(
                    Blur() if self.anti_alias else nn.Identity(),
                    conv2d(last_chan_small, last_chan_small, 3, 2, 1, bias=True),
                    batchNorm2d(last_chan_small), act,
                    conv2d(last_chan_small, 1, 4, 1, 0)
                )

        self.decoder_big = SimpleDecoder(channels[-1][-1], nc, use_tanh)
        if d_num_rec ==3:
            self.decoder_small = SimpleDecoder(channels[-2][-1], nc, use_tanh)
        self.decoder_part = SimpleDecoder(channels[-2][-1], nc, use_tanh)

    def forward(self, imgs, label):
        imgs_small = None
        if type(imgs) == list and len(imgs) == 1:
            imgs = imgs[0]
        elif type(imgs) == list and len(imgs) == 2:
            imgs, imgs_small = imgs

        x = self.down_from_big(imgs)

        layer_outputs = []
        for (res, (down, sle, attn)) in zip(self.non_residual_resolutions ,self.layers):
            residuals = dict()
            if exists(attn):
                x = attn(x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            x = down(x)
            # this method might require more GPU memory than before as it is storing the residuals
            next_res = res - 1
            if next_res in residuals:
                x = x * residuals[next_res] # combining skip-layer paths

            layer_outputs.append(x)

        if self.minibatch_d:
            out_std = torch.sqrt(x.var(0, unbiased=False)+1e-8)
            mean_std = out_std.mean()
            mean_std = mean_std.expand(x.size(0), 1, 8, 8)
            x = torch.cat([x, mean_std], 1)

        out_logits = self.to_logits(x).flatten(1)

        if self.multi_logits_out:
            if not exists(imgs_small):
                imgs_small = F.interpolate(imgs,size = self.small_im_size)
            feat_small = self.down_from_small(imgs_small)
            if self.minibatch_d:
                out_std = torch.sqrt(feat_small.var(0, unbiased=False)+1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(x.size(0), 1, 8, 8)
                feat_small = torch.cat([feat_small, mean_std], 1)

            out_logits_small = self.to_logits_small(feat_small).flatten(1)

        if label == 'real':
            rec_img_big = self.decoder_big(layer_outputs[-1])
            if self.d_num_rec == 3:
                rec_img_small = self.decoder_small(layer_outputs[-2]) # not in original paper
            part = random.randint(0, 3)
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(layer_outputs[-2][:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(layer_outputs[-2][:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(layer_outputs[-2][:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(layer_outputs[-2][:,:,8:,8:])

            if self.multi_logits_out:
                if self.d_num_rec == 3:
                    return torch.cat([out_logits, out_logits_small]), [rec_img_big, rec_img_small, rec_img_part], part
                else:
                    return torch.cat([out_logits, out_logits_small]), [rec_img_big, rec_img_part], part
            else:
                if self.d_num_rec == 3:
                    return out_logits, [rec_img_big, rec_img_small, rec_img_part], part
                else:
                    return out_logits, [rec_img_big, rec_img_part], part

        if self.multi_logits_out:
            return torch.cat([out_logits, out_logits_small])
        else:
            return out_logits


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3, use_tanh=True):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes*2), nn.GLU(dim=1))
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=True),
                                    nn.Tanh() if use_tanh else nn.Identity())

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

from random import randint
def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h-size-1)
    cw = randint(0, w-size-1)
    return image[:,:,ch:ch+size,cw:cw+size]
