'''
Adapted from https://github.com/lernapparat/lernapparat/tree/master/style_gan
To download network parameters for FFHQ and LSUN-Bedrooms, visit: https://drive.google.com/open?id=1GYzEzOCaI8FUS6JHdt6g9UfNTmpO08Tt
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import re
import numpy as np


class MyLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""
    def __init__(self, input_size, output_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size**(-0.5) # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class MyConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, gain=2 ** (0.5), use_wscale=False,
                 lrmul=1, bias=True,
                 intermediate=None, upscale=False, downscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, (1, 1, 1, 1))
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            w = F.pad(w, (1, 1, 1, 1))
            # in contrast to upscale, this is a mean...
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25  # avg_pool?
            x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noiselayers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise.to(x.device)
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size,
                            channels * 2,
                            gain=1.0, use_wscale=use_wscale)
        self.style = None

    def forward(self, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, style.shape[2]//2, 1, 1]# + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        return style



class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class BlurLayer(nn.Module):
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        #kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)

class Downscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x):
        assert x.dim()==4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        # Large factor => downscale using tf.nn.avg_pool().
        # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
        return F.avg_pool2d(x, self.factor)


class G_mapping(nn.Sequential):
    def __init__(self, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        layers = [
            ('pixel_norm', PixelNormLayer()),
            ('dense0', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense0_act', act),
            ('dense1', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense1_act', act),
            ('dense2', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense2_act', act),
            ('dense3', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense3_act', act),
            ('dense4', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense4_act', act),
            ('dense5', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense5_act', act),
            ('dense6', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense6_act', act),
            ('dense7', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense7_act', act)
        ]
        super().__init__(OrderedDict(layers))

    def forward(self, x):
        x = super().forward(x)
        # Broadcast
        x = x.unsqueeze(1).expand(-1, 18, -1)
        return x


class Truncation(nn.Module):
    def __init__(self, avg_latent=None, psi=1.0, latent_size=512):
        super().__init__()
        self.psi = psi
        self.register_buffer('avg_latent', avg_latent if avg_latent is not None else torch.zeros(latent_size))

    def forward(self, x):
        assert x.dim() == 3
        return torch.lerp(self.avg_latent, x, self.psi)


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(self, channels, use_noise, use_instance_norm, use_styles,
                 activation_layer):
        super().__init__()
        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', activation_layer))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))
        self.top_epi = nn.Sequential(OrderedDict(layers))
        self.use_styles = use_styles


    def forward(self, x, styles_in_slice=None):
        x = self.top_epi(x)

        if self.use_styles:
            x = x * (styles_in_slice[:, 0] + 1.) + styles_in_slice[:, 1] #apply style
        else:
            assert styles_in_slice is None
        return x



class InputBlock(nn.Module):
    def __init__(self, nf,  gain, use_wscale, use_noise,
                 use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.nf = nf

        self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
        self.bias = nn.Parameter(torch.ones(nf))

        self.epi1 = LayerEpilogue(nf, use_noise, use_instance_norm,
                                  use_styles, activation_layer)
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, use_noise, use_instance_norm,
                                  use_styles, activation_layer)

    def forward(self, styles_in_range):
        batch_size = styles_in_range[0].size(0)

        x = self.const.expand(batch_size, -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)

        x = self.epi1(x, styles_in_range[0])

        x = self.conv(x)
        x = self.epi2(x, styles_in_range[1])
        return x


class GSynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blur_filter, gain, use_wscale, use_noise,
                 use_instance_norm, use_styles, activation_layer):
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale,
                                 intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue(out_channels, use_noise, use_instance_norm,
                                  use_styles, activation_layer)
        self.conv1 = MyConv2d(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, use_noise, use_instance_norm,
                                  use_styles, activation_layer)

    def forward(self, x, styles_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, styles_in_range[0])
        x = self.conv1(x)
        x = self.epi2(x, styles_in_range[1])
        return x

class G_style(nn.Module):
    def __init__(self,
                 dlatent_size=512,  # Disentangled latent (W) dimensionality.
                 resolution=1024,  # Output resolution.
                 fmap_base=8192,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,  # Maximum number of feature maps in any layer.
                 use_wscale=True,  # Enable equalized learning rate?
                 ):

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        self.style_layers = nn.ModuleDict()

        for res in range(2, resolution_log2 + 1):
            channels = nf(res - 1)
            name = '{s}x{s}'.format(s=2 ** res)
            self.style_layers[name + '_0'] = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
            self.style_layers[name + '_1'] = StyleMod(dlatent_size, channels, use_wscale=use_wscale)

    def forward(self, w):
        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)
        styles = []
        for i, m in enumerate(self.style_layers.values()):
                styles.append(m(w[:, i:i+1]))
        return styles


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size=512,  # Disentangled latent (W) dimensionality.
                 num_channels=3,  # Number of output color channels.
                 resolution=1024,  # Output resolution.
                 fmap_base=8192,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,  # Maximum number of feature maps in any layer.
                 use_styles=True,  # Enable style inputs?
                 use_noise=True,  # Enable noise inputs?
                 nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu'
                 use_wscale=True,  # Enable equalized learning rate?
                 use_instance_norm=True,  # Enable instance normalization?
                 blur_filter=[1, 2, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
                 ):

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.resolution = resolution

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        num_layers = resolution_log2 - 2
        self.num_layers = num_layers


        blocks = []
        for res in range(2, resolution_log2 + 1):
            channels = nf(res - 1)
            name = '{s}x{s}'.format(s=2 ** res)
            if res == 2:
                blocks.append((name,
                               InputBlock(channels, gain, use_wscale,
                                          use_noise, use_instance_norm, use_styles, act)))

            else:
                blocks.append((name,
                               GSynthesisBlock(last_channels, channels, blur_filter, gain, use_wscale,
                                               use_noise, use_instance_norm, use_styles, act)))
            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale)
        self.blocks = nn.ModuleDict(OrderedDict(blocks))

        is_AdaIN = lambda item: type(item) is nn.InstanceNorm2d
        self.AdaIN_layers = list(filter(is_AdaIN, self.modules()))
        

    def forward(self, styles):
        for i, (res, m) in enumerate(self.blocks.items()):
            if i == 0:
                x = m(styles[2 * i:2 * i + 2])
            else:
                x = m(x, styles[2 * i:2 * i + 2])
        x = self.torgb(x)
        return x


    def set_noise(self, mode=None):
        assert mode is None or mode == 'zero' or mode == 'zeros' or mode == 'fixed'
        for k, m in self.blocks.named_children():
            res = tuple((int(sz) for sz in k.split('x')))
            if hasattr(m, 'conv'):
                device = m.conv.weight.device
                dtype = m.conv.weight.dtype
            elif hasattr(m, 'conv1'):
                device = m.conv1.weight.device
                dtype = m.conv1.weight.dtype
            else:
                raise Exception('Can not determine device and dtype for block', m)
            noise_layers = [l for l in m.modules() if type(l) is NoiseLayer]
            for nl in noise_layers:
                if mode is None:
                    nl.noise = None
                elif mode == 'zero' or mode == 'zeros':
                    nl.noise = torch.zeros(1, 1, res[0], res[1], device=device, dtype=dtype)
                else:  # fixed
                    nl.noise = torch.randn(1, 1, res[0], res[1], device=device, dtype=dtype)


class StyleGAN(nn.Module):

    @staticmethod
    def load_from_pth(filename):
        sd = torch.load(filename)
        max_res = max(
            [int(re.search('(\d+)x\d+', k).group(1)) for k in sd.keys() if k.endswith('conv1.weight')])
        G = StyleGAN( G_mapping(), Truncation(), G_style(resolution=max_res), G_synthesis(resolution=max_res))

        G.load_state_dict(sd)

        return G

    @property
    def AdaIN_layers(self):
        return self.g_synthesis.AdaIN_layers

    def __init__(self, g_mapping, truncation, g_style, g_synthesis):
        super().__init__()
        self.g_mapping = g_mapping
        self.truncation = truncation
        self.g_style = g_style
        self.g_synthesis = g_synthesis


    def z_to_w(self, z, truncation=1.0):
        self.truncation.psi = truncation
        return self.truncation(self.g_mapping(z))

    def w_to_ys(self, w):
        return self.g_style(w)

    def ys_to_rgb(self, ys):
        return self.g_synthesis(ys)

    def z_to_ys(self, z, truncation=1.0):
        return self.w_to_ys(self.z_to_w(z, truncation=truncation))

    def forward(self, z, truncation=1.0):
        return self.ys_to_rgb(self.w_to_ys(self.z_to_w(z, truncation=truncation)))

