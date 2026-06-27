# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw

# ===================== 🔥 新增：脊柱神经专用 Attention Gate 模块 =====================
class AttentionGate(BaseModule):
    """
    针对 1/4 和 1/8 高分辨率层级的注意力门控。
    使用深层特征 (g) 指导浅层特征 (x) 的提取。
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: 来自深层的特征 (低分辨率)
        # x: 来自浅层的跳跃连接特征 (高分辨率)
        
        # 将深层特征上采样到浅层尺寸
        g1 = self.W_g(F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False))
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi # 返回加权后的浅层特征

# =================================================================================

class MixFFN(BaseModule):
    # ... (保持原版 MixFFN 不变)
    def __init__(self, embed_dims, feedforward_channels, act_cfg=dict(type='GELU'),
                 ffn_drop=0., dropout_layer=None, init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        in_channels = embed_dims
        fc1 = Conv2d(in_channels=in_channels, out_channels=feedforward_channels, kernel_size=1, stride=1, bias=True)
        pe_conv = Conv2d(in_channels=feedforward_channels, out_channels=feedforward_channels, kernel_size=3, stride=1,
                         padding=(3 - 1) // 2, bias=True, groups=feedforward_channels)
        fc2 = Conv2d(in_channels=feedforward_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None: identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    # ... (保持原版 EfficientMultiheadAttention 不变)
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_layer=None,
                 init_cfg=None, batch_first=True, qkv_bias=False, norm_cfg=dict(type='LN'), sr_ratio=1):
        super().__init__(embed_dims, num_heads, attn_drop, proj_drop, dropout_layer=dropout_layer,
                         init_cfg=init_cfg, batch_first=batch_first, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape); x_kv = self.sr(x_kv); x_kv = nchw_to_nlc(x_kv); x_kv = self.norm(x_kv)
        else: x_kv = x
        if identity is None: identity = x_q
        if self.batch_first:
            x_q = x_q.transpose(0, 1); x_kv = x_kv.transpose(0, 1)
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        if self.batch_first: out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    # ... (保持原版 TransformerEncoderLayer 不变)
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., qkv_bias=True, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'),
                 batch_first=True, sr_ratio=1, with_cp=False):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = EfficientMultiheadAttention(embed_dims=embed_dims, num_heads=num_heads, attn_drop=attn_drop_rate,
                                               proj_drop=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                                               batch_first=batch_first, qkv_bias=qkv_bias, norm_cfg=norm_cfg, sr_ratio=sr_ratio)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = MixFFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, ffn_drop=drop_rate,
                          dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate), act_cfg=act_cfg)
        self.with_cp = with_cp

    def forward(self, x, hw_shape):
        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x
        if self.with_cp and x.requires_grad: x = cp.checkpoint(_inner_forward, x)
        else: x = _inner_forward(x)
        return x


@MODELS.register_module()
class MixVisionTransformerMulti(BaseModule):
    def __init__(self, in_channels=3, embed_dims=64, num_stages=4, num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6), pretrained=None, init_cfg=None,
                 with_cp=False, **kwargs):
        super().__init__(init_cfg=init_cfg)

        # ... (参数初始化部分保持不变)
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.out_indices = out_indices
        encoder_params = kwargs.get('encoder_params', {'interval': 2})
        self.interval = encoder_params['interval']

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(in_channels=in_channels, embed_dims=embed_dims_i, kernel_size=patch_sizes[i],
                                    stride=strides[i], padding=patch_sizes[i] // 2, norm_cfg=norm_cfg)
            layer = ModuleList([
                TransformerEncoderLayer(embed_dims=embed_dims_i, num_heads=num_heads[i],
                                       feedforward_channels=mlp_ratio * embed_dims_i, drop_rate=drop_rate,
                                       attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[cur + idx],
                                       qkv_bias=qkv_bias, act_cfg=act_cfg, norm_cfg=norm_cfg,
                                       with_cp=with_cp, sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        # 🔥 新增：初始化 Attention Gates (只针对 Stage 1 和 Stage 2)
        # Stage 1 (1/4) 和 Stage 2 (1/8)
        # 注意力来自深层 Stage 4 (1/32)
        self.ag1 = AttentionGate(F_g=embed_dims*num_heads[3], F_l=embed_dims*num_heads[0], F_int=embed_dims*num_heads[0])
        self.ag2 = AttentionGate(F_g=embed_dims*num_heads[3], F_l=embed_dims*num_heads[1], F_int=embed_dims*num_heads[1])

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            outs.append(x)

        # 🔥 关键修改：在输出前应用 Attention Gate
        # 我们用最后一层 outs[3] (Stage 4) 作为 Gate 信号引导 outs[0] 和 outs[1]
        g = outs[3]
        
        # 仅针对最后两个高分辨率层级 (Stage 1 和 Stage 2) 做处理
        refined_out0 = self.ag1(g, outs[0]) # 1/4 尺度
        refined_out1 = self.ag2(g, outs[1]) # 1/8 尺度
        
        # 返回最终特征组
        return (refined_out0, refined_out1, outs[2], outs[3])