# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw

# ===================== 🔥 高级增强模块：MS-TFA (针对细长神经拓扑增强) =====================
class MSTFA_Module(BaseModule):
    """
    多尺度拓扑特征聚合模块
    结合了 Strip Pooling (条形池化) 和 局部精修，专门解决脊柱神经断裂问题
    """
    def __init__(self, in_channels):
        super().__init__()
        inter_channels = in_channels // 2
        
        # 1. 条形池化：捕捉水平和垂直的长距离依赖 (神经的典型几何特征)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv_h = nn.Conv2d(in_channels, inter_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv_w = nn.Conv2d(in_channels, inter_channels, kernel_size=(1, 3), padding=(0, 1))
        
        # 2. 局部精修分支：保持边缘细节
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, groups=inter_channels),
            nn.BatchNorm2d(inter_channels),
            nn.GELU()
        )
        
        # 3. 动态融合层
        self.out_conv = nn.Sequential(
            nn.Conv2d(inter_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        h, w = x.shape[2:]
        
        # 条形特征提取
        x_h = self.conv_h(self.pool_h(x).expand(-1, -1, h, w))
        x_w = self.conv_w(self.pool_w(x).expand(-1, -1, h, w))
        x_l = self.refine(x)
        
        # 空间注意力加权
        att = self.out_conv(torch.cat([x_h, x_w, x_l], dim=1))
        return identity * att

# =====================================================================================

class MixFFN(BaseModule):
    def __init__(self, embed_dims, feedforward_channels, act_cfg=dict(type='GELU'), ffn_drop=0., dropout_layer=None):
        super().__init__()
        self.activate = build_activation_layer(act_cfg)
        self.layers = Sequential(
            Conv2d(embed_dims, feedforward_channels, 1),
            Conv2d(feedforward_channels, feedforward_channels, 3, padding=1, groups=feedforward_channels),
            self.activate,
            nn.Dropout(ffn_drop),
            Conv2d(feedforward_channels, embed_dims, 1),
            nn.Dropout(ffn_drop)
        )
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        return (x if identity is None else identity) + self.dropout_layer(out)

class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_layer=None, 
                 batch_first=True, qkv_bias=False, norm_cfg=dict(type='LN'), sr_ratio=1):
        super().__init__(embed_dims, num_heads, attn_drop, proj_drop, dropout_layer=dropout_layer, 
                         batch_first=batch_first, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(embed_dims, embed_dims, sr_ratio, stride=sr_ratio)
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else: x_kv = x
        if self.batch_first:
            x_q, x_kv = x_q.transpose(0, 1), x_kv.transpose(0, 1)
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        if self.batch_first: out = out.transpose(0, 1)
        return (x if identity is None else identity) + self.dropout_layer(self.proj_drop(out))

class TransformerEncoderLayer(BaseModule):
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., qkv_bias=True, act_cfg=dict(type='GELU'), 
                 norm_cfg=dict(type='LN'), sr_ratio=1):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = EfficientMultiheadAttention(embed_dims, num_heads, attn_drop_rate, drop_rate, 
                                               dict(type='DropPath', drop_prob=drop_path_rate), 
                                               True, qkv_bias, norm_cfg, sr_ratio)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = MixFFN(embed_dims, feedforward_channels, act_cfg, drop_rate, 
                          dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x

@MODELS.register_module()
class MixVisionTransformerMulti(BaseModule):
    def __init__(self, in_channels=3, embed_dims=64, num_stages=4, num_layers=[3, 4, 6, 3], 
                 num_heads=[1, 2, 4, 8], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], 
                 sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True, 
                 drop_rate=0., drop_path_rate=0., norm_cfg=dict(type='LN', eps=1e-6), **kwargs):
        super().__init__()
        
        self.out_indices = out_indices
        encoder_params = kwargs.get('encoder_params', {'interval': 2})
        self.interval = encoder_params['interval']
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
        cur = 0
        self.layers = ModuleList()
        
        for i in range(num_stages):
            dim = embed_dims * num_heads[i]
            # ✅ 修复关键点：使用显式关键字传参防止 Registry.get 报错
            patch_embed = PatchEmbed(
                in_channels=in_channels if i == 0 else embed_dims * num_heads[i - 1],
                embed_dims=dim,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg
            )
            
            blocks = ModuleList([
                TransformerEncoderLayer(dim, num_heads[i], mlp_ratio * dim, drop_rate, 0., 
                                       dpr[cur + j], qkv_bias, dict(type='GELU'), norm_cfg, 
                                       sr_ratios[i]) for j in range(num_layers[i])
            ])
            norm = build_norm_layer(norm_cfg, dim)[1]
            self.layers.append(ModuleList([patch_embed, blocks, norm]))
            cur += num_layers[i]

        # 🔥 高级增强：在最后两个高分辨率阶段（1/4, 1/8）注入 MS-TFA
        self.enhance1 = MSTFA_Module(embed_dims * num_heads[0])
        self.enhance2 = MSTFA_Module(embed_dims * num_heads[1])

    def forward(self, x):
        outs = []
        mid_outs = [] # 保持原版输出逻辑兼容
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x) # Patch-Embed
            for blk_n, block in enumerate(layer[1]):
                x = block(x, hw_shape)
                # 兼容原版的间隔输出逻辑
                if i == 2 and blk_n != 0 and blk_n % self.interval == 0:
                    mid_outs.append(nlc_to_nchw(x, hw_shape))
            
            x = layer[2](x) # Norm
            x = nlc_to_nchw(x, hw_shape)
            
            # 执行 MS-TFA 增强逻辑
            if i == 0: x = self.enhance1(x)
            if i == 1: x = self.enhance2(x)
            
            if i in self.out_indices:
                outs.append(x)
                
        return tuple(outs)