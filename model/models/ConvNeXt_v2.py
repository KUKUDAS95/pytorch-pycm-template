""" ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
- Papaer: https://arxiv.org/pdf/2301.00808v1.pdf
- Official Code: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py

- The models used in the paper are all.
"""

from base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_url_checkpoint, change_kwargs # for pre-trained model

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(BaseModel): #(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _change_model_num_class(model, num_classes):
    model.head = nn.Linear(model.head.in_features, num_classes, bias=model.head.bias.requires_grad)
    return model
def _change_model_in_chans(model, in_chans):
    model.downsample_layers[0][0] = nn.Conv2d(in_chans, model.downsample_layers[0][0].out_channels,
                                              kernel_size=model.downsample_layers[0][0].kernel_size, 
                                              stride=model.downsample_layers[0][0].stride) 
    return model

def convnextv2_atto(pretrained:bool=False, **kwargs):
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    
    if pretrained:
        model.load_state_dict(load_url_checkpoint('https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt'))
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
        if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans) 
    return model

def convnextv2_femto(pretrained:bool=False, **kwargs):
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    
    if pretrained:
        model.load_state_dict(load_url_checkpoint('https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt'))
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
        if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans) 
    return model

def convnextv2_pico(pretrained:bool=False, **kwargs):
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    
    if pretrained:
        model.load_state_dict(load_url_checkpoint('https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt'))
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
        if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans) 
    return model

def convnextv2_nano(pretrained:bool=False, **kwargs):
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    
    if pretrained:
        model.load_state_dict(load_url_checkpoint('https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt'))
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
        if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans) 
    return model

def convnextv2_tiny(pretrained:bool=False, **kwargs):
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    
    if pretrained:
        model.load_state_dict(load_url_checkpoint('https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt'))
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
        if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans) 
    return model

def convnextv2_base(pretrained:bool=False, **kwargs):
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    
    if pretrained:
        model.load_state_dict(load_url_checkpoint('https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt'))
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
        if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans) 
    return model

def convnextv2_large(pretrained:bool=False, **kwargs):
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    
    if pretrained:
        model.load_state_dict(load_url_checkpoint('https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt'))
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
        if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans) 
    return model

def convnextv2_huge(pretrained:bool=False, **kwargs):
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    
    if pretrained:
        model.load_state_dict(load_url_checkpoint('https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt'))
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
        if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans) 
    return model