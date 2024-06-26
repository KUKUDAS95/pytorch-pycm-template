""" DeiT: Data-efficient Image Transformers
- Papaer: https://arxiv.org/pdf/2012.12877.pdf
- Official Code: https://github.com/facebookresearch/deit/blob/main/models_v2.py

- The models used in the paper are DeiT-tiny, DeiT-small, and DeiT-base.
    - DeiT-tiny  : embedding_dim 192, heads 3,  layers 12 -> deit_tiny_patch16_LS
    - DeiT-small : embedding_dim 384, heads 6,  layers 12 -> deit_small_patch16_LS
    - DeiT-base  : embedding_dim 768, heads 12, layers 12 -> deit_base_patch16_LS
"""

""" DeiT III: Revenge of the ViT
- Papaer: https://arxiv.org/pdf/2204.07118v1.pdf
- Official Code: https://github.com/facebookresearch/deit/blob/main/models_v2.py

- The models used in the paper are ViT-S, ViT-B, ViT-L, and ViT-H.
    - ViT-S      : embedding_dim 384,  heads 6,  layers 12 -> deit_small_patch16_LS
    - ViT-B      : embedding_dim 768,  heads 12, layers 12 -> deit_base_patch16_LS
    - ViT-L      : embedding_dim 1024, heads 16, layers 24 -> deit_large_patch16_LS
    - ViT-H      : embedding_dim 1280, heads 16, layers 32 -> deit_huge_patch14_LS
- They say "In this paper, we revisit the supervised training of ViTs. Our procedure builds upon and simplifies a recipe introduced for training ResNet-50."
    - 3-Augment: We propose a simple data augmentation inspired by what is used in self-supervised learning (SSL).
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from base import BaseModel

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from utils import change_kwargs, load_url_checkpoint  # for pre-trained model

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        
    def forward(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x
        
class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x
        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size=img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class vit_models(BaseModel): #(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,**kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x


def _change_model_num_class(model, num_classes):
    model.head = nn.Linear(model.head.in_features, num_classes, bias=model.head.bias.requires_grad)
    return model
def _change_model_in_chans(model, in_chans):
    model.patch_embed.proj = nn.Conv2d(in_chans, model.patch_embed.proj.out_channels, 
                                       kernel_size=model.patch_embed.proj.kernel_size, stride=model.patch_embed.proj.stride)
    return model


# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)
@register_model
def deit_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    # DeiT: Data-efficient Image Transformers
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    
    return model
    
    
@register_model
def deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    # DeiT: Data-efficient Image Transformers
    # DeiT III: Revenge of the ViT
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        path = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_224_' + ('21k.pth' if pretrained_21k else '1k.pth')
        model.load_state_dict(load_url_checkpoint(path))
        if img_size != 224:   
            if in_chans is None: in_chans = 3
            Patch_layer = PatchEmbed if 'Patch_layer' not in list(kwargs.keys()) else kwargs['Patch_layer']
            new_patch_embed = Patch_layer(img_size=img_size, patch_size=16, in_chans=in_chans, embed_dim=384)
            num_patches = new_patch_embed.num_patches
            new_pos_embed = nn.Parameter(torch.zeros(1, num_patches, 384))
            trunc_normal_(new_pos_embed, std=.02)
            
            model.patch_embed = new_patch_embed
            model.pos_embed = new_pos_embed
        else:
            if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans)
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
         
    return model

@register_model
def deit_medium_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers = Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        path = 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_' + ('21k.pth' if pretrained_21k else '1k.pth')
        model.load_state_dict(load_url_checkpoint(path))
        if img_size != 224:   
            if in_chans is None: in_chans = 3
            Patch_layer = PatchEmbed if 'Patch_layer' not in list(kwargs.keys()) else kwargs['Patch_layer']
            new_patch_embed = Patch_layer(img_size=img_size, patch_size=16, in_chans=in_chans, embed_dim=384)
            num_patches = new_patch_embed.num_patches
            new_pos_embed = nn.Parameter(torch.zeros(1, num_patches, 384))
            trunc_normal_(new_pos_embed, std=.02)
            
            model.patch_embed = new_patch_embed
            model.pos_embed = new_pos_embed
        else:
            if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans)
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
    return model 

@register_model
def deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    # DeiT: Data-efficient Image Transformers
    # DeiT III: Revenge of the ViT
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        path = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_224_' + ('21k.pth' if pretrained_21k else '1k.pth')
        model.load_state_dict(load_url_checkpoint(path))
        if img_size != 224:   
            if in_chans is None: in_chans = 3
            Patch_layer = PatchEmbed if 'Patch_layer' not in list(kwargs.keys()) else kwargs['Patch_layer']
            new_patch_embed = Patch_layer(img_size=img_size, patch_size=16, in_chans=in_chans, embed_dim=384)
            num_patches = new_patch_embed.num_patches
            new_pos_embed = nn.Parameter(torch.zeros(1, num_patches, 384))
            trunc_normal_(new_pos_embed, std=.02)
            
            model.patch_embed = new_patch_embed
            model.pos_embed = new_pos_embed
        else:
            if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans)
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
    return model
    
@register_model
def deit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    # DeiT III: Revenge of the ViT
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        path = 'https://dl.fbaipublicfiles.com/deit/deit_3_large_224_' + ('21k.pth' if pretrained_21k else '1k.pth')
        model.load_state_dict(load_url_checkpoint(path))
        if img_size != 224:   
            if in_chans is None: in_chans = 3
            Patch_layer = PatchEmbed if 'Patch_layer' not in list(kwargs.keys()) else kwargs['Patch_layer']
            new_patch_embed = Patch_layer(img_size=img_size, patch_size=16, in_chans=in_chans, embed_dim=384)
            num_patches = new_patch_embed.num_patches
            new_pos_embed = nn.Parameter(torch.zeros(1, num_patches, 384))
            trunc_normal_(new_pos_embed, std=.02)
            
            model.patch_embed = new_patch_embed
            model.pos_embed = new_pos_embed
        else:
            if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans)
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
    return model
    
@register_model
def deit_huge_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    # DeiT III: Revenge of the ViT
    if pretrained: num_classes, in_chans, kwargs = change_kwargs(**kwargs)
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers = Layer_scale_init_Block, **kwargs)
    if pretrained:            
        path = 'https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_' + ('21k_v1.pth' if pretrained_21k else '1k_v1.pth')
        model.load_state_dict(load_url_checkpoint(path))
        if img_size != 224:   
            if in_chans is None: in_chans = 3
            Patch_layer = PatchEmbed if 'Patch_layer' not in list(kwargs.keys()) else kwargs['Patch_layer']
            new_patch_embed = Patch_layer(img_size=img_size, patch_size=16, in_chans=in_chans, embed_dim=384)
            num_patches = new_patch_embed.num_patches
            new_pos_embed = nn.Parameter(torch.zeros(1, num_patches, 384))
            trunc_normal_(new_pos_embed, std=.02)
            
            model.patch_embed = new_patch_embed
            model.pos_embed = new_pos_embed
        else:
            if in_chans is not None and in_chans != 3: model = _change_model_in_chans(model, in_chans)
        if num_classes is not None and num_classes != 1000: model = _change_model_num_class(model, num_classes)
    return model
    
@register_model
def deit_huge_patch14_52_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=14, embed_dim=1280, depth=52, num_heads=16, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers = Layer_scale_init_Block, **kwargs)

    return model
    
@register_model
def deit_huge_patch14_26x2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=14, embed_dim=1280, depth=26, num_heads=16, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers = Layer_scale_init_Block_paralx2, **kwargs)

    return model
    
""" Block_paral_LS is not defined
@register_model
def deit_Giant_48x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size = 224 if pretrained else img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers = Block_paral_LS, **kwargs)

    return model

@register_model
def deit_giant_40x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size = 224 if pretrained else img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers = Block_paral_LS, **kwargs)
    return model
"""

@register_model
def deit_Giant_48_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers = Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_giant_40_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers = Layer_scale_init_Block, **kwargs)
    return model

# Models from Three things everyone should know about Vision Transformers (https://arxiv.org/pdf/2203.09795.pdf)
@register_model
def deit_small_patch16_36_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model
    
@register_model
def deit_small_patch16_36(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
@register_model
def deit_small_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)
    return model
    
@register_model
def deit_small_patch16_18x2(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Block_paralx2, **kwargs)
    return model
    
  
@register_model
def deit_base_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)
    return model


@register_model
def deit_base_patch16_18x2(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Block_paralx2, **kwargs)
    return model
    

@register_model
def deit_base_patch16_36x1_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_base_patch16_36x1(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    if pretrained: raise ValueError('This module does not include pre-trained models. If you have the file, you can modify that part.')
    model = vit_models(img_size = 224 if pretrained else img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, 
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
