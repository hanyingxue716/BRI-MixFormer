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

# ===================== 🔥 高级增强：PDRM (动态残差感知模块) =====================
class PDRM_Module(BaseModule):
    """
    针对手术场景优化的动态增强模块：
    1. 动态权重：自动适应椭圆、长条等碎片化神经
    2. 抗干扰：通过通道门控抑制血雾与反光
    3. 稳定：初始化为 0，确保性能只增不减
    """
    def __init__(self, in_channels):
        super().__init__()
        # 深度可分离卷积：在不增加过多参数的情况下，增强局部异形目标的捕捉
        self.branch_local = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
        # 针对血雾和强光的全局上下文门控
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 关键：可学习的缩放因子，初始化为 0。
        # 它的作用是让模型先继承原始性能，再根据数据缓慢学习增强逻辑。
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x
        # 1. 捕捉不同形态的碎片特征
        feat = self.branch_local(x)
        # 2. 生成抗干扰权重图
        gate = self.context_gate(x)
        # 3. 动态融合：x + gamma * (feat * gate)
        return identity + self.gamma * (feat * gate)

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

        # 🔥 只在 Stage 1 (1/8 尺度) 和 Stage 2 (1/16 尺度) 加入增强
        # 1/4 尺度背景杂质太多，1/32 尺度分辨率太低，中间层是区分裸露碎片的最优选择
        self.enhance1 = PDRM_Module(embed_dims * num_heads[1])
        self.enhance2 = PDRM_Module(embed_dims * num_heads[2])

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for blk_n, block in enumerate(layer[1]):
                x = block(x, hw_shape)
            
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            
            # 执行 PDRM 增强
            if i == 1: x = self.enhance1(x)
            if i == 2: x = self.enhance2(x)
            
            if i in self.out_indices:
                outs.append(x)
                
        return tuple(outs)