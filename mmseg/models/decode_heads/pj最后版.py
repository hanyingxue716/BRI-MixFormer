# ---------------------------------------------------------------
# Copyright (c) 2021, Nota AI GmbH. All rights reserved.原始版本！！
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from ..utils import resize
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import *
import math
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F

# ... [DWConv, Mlp, CatKey, CatKeyMulti, CrossAttention, Block 等类保持不变] ...
# [此处省略你代码中前面的辅助类定义，直接进入 Head 的修改]

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class CatKey(nn.Module):
    def __init__(self, pool_ratio=[1,2,4,8], dim=[256,160,64,32]):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = nn.ModuleList([nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])
        self.pool_list = nn.ModuleList([nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])

    def forward(self, x):
        out_list = []
        cnt = 0
        for i in range(len(self.pool_ratio)):
            if self.pool_ratio[i] > 1:
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](x[i])))
                cnt += 1
            else:
                out_list.append(x[i])
        return torch.cat(out_list, dim=1)

class CatKeyMulti(nn.Module):
    def __init__(self, pool_ratio=[1,2,4,8], dim=[256,160,64,32], num_feat = 4):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = nn.ModuleList([nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])
        self.pool_list = nn.ModuleList([nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True) for i in range(len(self.pool_ratio)) if self.pool_ratio[i] > 1])
        for _ in range(num_feat):
            self.sr_list.append(nn.Conv2d(dim[1], dim[1], kernel_size=1, stride=1))
            self.pool_list.append(nn.AvgPool2d(self.pool_ratio[1], self.pool_ratio[1], ceil_mode=True))

    def forward(self, x, xMulti):
        out_list = []
        cnt = 0
        for i in range(len(self.pool_ratio)):
            if self.pool_ratio[i] > 1:
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](x[i])))
                cnt += 1
            else:
                out_list.append(x[i])
        for l in range(len(xMulti)):
            out_list.append(self.sr_list[cnt](self.pool_list[cnt](xMulti[l])))
            cnt += 1
        return torch.cat(out_list, dim=1)

class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."
        self.dim1, self.dim2, self.num_heads = dim1, dim2, num_heads
        head_dim = dim1 // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        x_ = self.act(self.norm(y)) if self.pool_ratio >= 0 else y
        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B1, N1, C1)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2, H1, W1):
        x = self.norm1(x)
        y = self.norm2(y)
        x = x + self.drop_path(self.attn(x, y, H2, W2))
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, H1, W1))
        return x
class RPDA_CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        
        # --- 创新点：Delta Rule 专属参数 ---
        # 用于将 Query 逆投影回 Key 空间，计算信息残差
        self.k_rec = nn.Linear(head_dim, head_dim, bias=False) 
        # 用于将 Key 的误差转换到 Value 的补偿空间
        self.v_err = nn.Linear(head_dim, head_dim, bias=False) 
        # 可学习的误差补偿权重，初始化为0，保证训练初期的稳定性 (Identity mapping)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        # -----------------------------------

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.pool_ratio >= 0:
            self.pool = nn.AvgPool2d(self.pool_ratio, self.pool_ratio)
            self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        
        # [B, H, N1, C_h]
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        if self.pool_ratio >= 0:
            x_ = self.norm(y)
            x_ = self.act(x_)  # 修复了原版的 bug：原版这里是 self.act(y)
        else:
            x_ = y
            
        # [2, B, H, N2, C_h]
        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # --- 1. Primary Attention (主注意力) ---
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N1, N2]
        attn_soft = attn.softmax(dim=-1)
        attn_dropped = self.attn_drop(attn_soft)

        v_out = (attn_dropped @ v) # [B, H, N1, C_h]

        # --- 2. 创新点：Retro-Projective Delta Rule (RPDA 逆投影误差修正) ---
        # 步骤 2.1: 逆投影。A^T @ Q -> 得到“被 Query 成功匹配到的 Key 信息”
        k_hat = (attn_soft.transpose(-2, -1) @ q)  # [B, H, N2, C_h]
        k_hat = self.k_rec(k_hat)

        # 步骤 2.2: Delta Rule 计算未解释的残差 (Unexplained Key Error)
        e_k = k - k_hat

        # 步骤 2.3: 误差补偿投影。将 Key 误差投影为 Value 的补偿量
        e_v = self.v_err(e_k)

        # 步骤 2.4: 误差二次重组。复用原注意力矩阵，将丢失的特征定向补偿给 Query
        v_comp = (attn_dropped @ e_v)

        # 步骤 2.5: 动态门控融合。
        v_final = v_out + self.gamma * v_comp
        # -------------------------------------------------------------

        # Output Projection
        x_out = v_final.transpose(1, 2).reshape(B1, N1, C1)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class TextureEdgeOcclusionGate(nn.Module):
    """
    纹理-边缘联合门控模块 (Texture-Edge Occlusion Gate)
    专门用于处理手术影像中边缘不清晰和病灶区域纹理特征提取。
    """
    def __init__(self, in_channels, num_classes):
        super(TextureEdgeOcclusionGate, self).__init__()
        
        # 1. 纹理分支：通过深层空洞卷积/分组卷积捕获纹理细节
        self.texture_branch = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='GELU')
        )
        
        # 2. 边缘门控分支：通过空间注意力定位边缘遮挡区域
        self.edge_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. 融合洗练层
        self.fusion_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='GELU')
        )
        
        # 4. 最终预测头
        self.cls_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # 提取纹理特征
        tex_feat = self.texture_branch(x)
        
        # 生成边缘掩码/置信度图
        edge_mask = self.edge_gate(x)
        
        # 门控融合：使用边缘掩码调节特征分布
        # 强化边缘对比度，抑制遮挡噪声
        refined_feat = x * edge_mask + tex_feat * (1 - edge_mask)
        
        # 进一步融合
        out_feat = self.fusion_conv(refined_feat)
        
        # 生成分割预测
        logits = self.cls_seg(out_feat)
        
        # 返回 logits 和 边缘图（供调试/可视化，虽然主逻辑里只取 logits）
        return logits, edge_mask

# ======================== 完整 Head 定义 ========================
@MODELS.register_module()
class APFormerHead_ORT(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead_ORT, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.dropout = nn.Dropout2d(0.1)
        self.ort_head = TextureEdgeOcclusionGate(embedding_dim, self.num_classes)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x

        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c4_t = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4_t, c_key, h4, w4, h4, w4)

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        c_key = self.cat_key2([_c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c3_t = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_t, c_key, h4, w4, h3, w3)

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3([_c4, _c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c2_t = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_t, c_key, h4, w4, h2, w2)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4([_c4, _c3, _c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c1_t = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1_t, c_key, h4, w4, h1, w1)

        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

        fused = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        fused = self.dropout(fused)
        
        seg_logits, conf_map = self.ort_head(fused)

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
            
        seg_logits = resize(seg_logits, size=target_size, mode='bilinear', align_corners=self.align_corners)
        
        # ✅ 【关键修复】MMSegmentation 的 loss 计算流要求 forward 必须返回 Tensor
        # conf_map 仅用于内部计算，不向上暴露，避免破坏标准 pipeline
        return seg_logits
        
# ======================== 适配 RPDA 的 Block 封装 ========================
class RPDA_Block(nn.Module):
    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., drop_path=0., pool_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim1)
        self.attn = RPDA_CrossAttention(dim1, dim2, num_heads=num_heads, pool_ratio=pool_ratio)
        
        # 为了避免依赖外部库的 DropPath，这里做安全回退
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim1, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_path),
            nn.Linear(mlp_hidden_dim, dim1),
            nn.Dropout(drop_path)
        )

    def forward(self, x, y, H1, W1, H2, W2):
        # 典型的 Transformer Block 结构
        x = x + self.drop_path(self.attn(self.norm1(x), y, H2, W2))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.GELU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_w * a_h
class MultiScaleRefine(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c3 = ConvModule(in_c, in_c, 3, padding=1, groups=in_c, act_cfg=dict(type='GELU'))
        self.c5 = ConvModule(in_c, in_c, 5, padding=2, groups=in_c, act_cfg=dict(type='GELU'))
        self.fuse = nn.Conv2d(in_c * 2, in_c, 1)
        # 建议：如果你的 batch_size 很小，可以把这里的 BatchNorm2d 换成 SyncBatchNorm
        self.norm = nn.BatchNorm2d(in_c)

    def forward(self, x):
        return x + F.gelu(self.norm(self.fuse(torch.cat([self.c3(x), self.c5(x)], dim=1))))


# ======================== 2. 核心解码头 APFormerHead_SpinePro5 ========================
@MODELS.register_module()
class APFormerHead_SpinePro5(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead_SpinePro5, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        # 注意力 Block (保持你原始代码中的 Block 类调用)
        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        # 【新增】为每个层级初始化多尺度细化模块
        self.refine_c4 = MultiScaleRefine(c4_in_channels)
        self.refine_c3 = MultiScaleRefine(c3_in_channels)
        self.refine_c2 = MultiScaleRefine(c2_in_channels)
        self.refine_c1 = MultiScaleRefine(c1_in_channels)

        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # --- Stage 4 ---
        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c4_t = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4_t, c_key, h4, w4, h4, w4)
        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        _c4 = self.refine_c4(_c4) # 【注入多尺度洗练】

        # --- Stage 3 ---
        c_key = self.cat_key2([_c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c3_t = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_t, c_key, h4, w4, h3, w3)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        _c3 = self.refine_c3(_c3) # 【注入多尺度洗练】

        # --- Stage 2 ---
        c_key = self.cat_key3([_c4, _c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c2_t = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_t, c_key, h4, w4, h2, w2)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        _c2 = self.refine_c2(_c2) # 【注入多尺度洗练】

        # --- Stage 1 ---
        c_key = self.cat_key4([_c4, _c3, _c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c1_t = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1_t, c_key, h4, w4, h1, w1)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)
        _c1 = self.refine_c1(_c1) # 【注入多尺度洗练】

        # 特征对齐与拼接
        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        # 最终预测与尺寸还原
        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
            
        x = resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
        return x
@MODELS.register_module()
class APFormerHead_SpinePro_Final(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead_SpinePro_Final, self).__init__(input_transform='multiple_select', **kwargs)
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        # --- 核心组件 ---
        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        # --- 深度监督辅助头 ---
        self.ds_head_c4 = nn.Conv2d(c4_in_channels, self.num_classes, kernel_size=1)
        self.ds_head_c3 = nn.Conv2d(c3_in_channels, self.num_classes, kernel_size=1)
        self.ds_head_c2 = nn.Conv2d(c2_in_channels, self.num_classes, kernel_size=1)

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # --- 三合一损失函数实例化 ---
        self.bfd_loss = BoundaryFocalDiceLoss(ignore_index=self.ignore_index)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        h1, w1 = c1.shape[2:]

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)

        # Stage 处理
        c_key = self.cat_key1([c4, c3, c2, c1]).flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4.flatten(2).transpose(1, 2), c_key, h4, w4, h4, w4).permute(0,2,1).reshape(n, -1, h4, w4)

        c_key = self.cat_key2([_c4, c3, c2, c1]).flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3.flatten(2).transpose(1, 2), c_key, h4, w4, c3.shape[2], c3.shape[3]).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])

        c_key = self.cat_key3([_c4, _c3, c2, c1]).flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2.flatten(2).transpose(1, 2), c_key, h4, w4, c2.shape[2], c2.shape[3]).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])

        c_key = self.cat_key4([_c4, _c3, _c2, c1]).flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1.flatten(2).transpose(1, 2), c_key, h4, w4, h1, w1).permute(0,2,1).reshape(n, -1, h1, w1)

        # 训练模式：计算深度监督输出
        if self.training:
            out_ds4 = resize(self.ds_head_c4(_c4), size=target_size, mode='bilinear', align_corners=self.align_corners)
            out_ds3 = resize(self.ds_head_c3(_c3), size=target_size, mode='bilinear', align_corners=self.align_corners)
            out_ds2 = resize(self.ds_head_c2(_c2), size=target_size, mode='bilinear', align_corners=self.align_corners)

        # 主预测路径
        _c4_up = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3_up = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2_up = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)

        _fuse = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1], dim=1))
        x_main = self.linear_pred(self.dropout(_fuse))
        x_main = resize(x_main, size=target_size, mode='bilinear', align_corners=self.align_corners)

        if self.training:
            return [x_main, out_ds4, out_ds3, out_ds2]
        else:
            return x_main

    def loss_by_feat(self, seg_logits, batch_data_samples):
        """核心修复：协调多个预测头与三合一损失"""
        loss = dict()
        seg_label = torch.stack([d.gt_sem_seg.data for d in batch_data_samples], dim=0).squeeze(1).long()
        
        if isinstance(seg_logits, list):
            # 1. 计算主头损失 (权重 1.0)
            loss['loss_bfd_main'] = self.bfd_loss(seg_logits[0], seg_label)

            # 2. 计算深度监督辅助头损失 (权重建议设为 0.4)
            for i, logits in enumerate(seg_logits[1:]):
                loss[f'loss_bfd_ds_{i}'] = self.bfd_loss(logits, seg_label) * 0.4
                
            # 3. 计算辅助精度指标 (仅观察主预测头)
            with torch.no_grad():
                pred = seg_logits[0].argmax(1)
                mask = seg_label != self.ignore_index
                acc = ((pred == seg_label) & mask).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0, device=pred.device)
                loss['acc_seg'] = acc
        else:
            # 推理模式
            loss['loss_bfd'] = self.bfd_loss(seg_logits, seg_label)
            
        return loss
@MODELS.register_module()
class APFormerHead_SpinePro7(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead_SpinePro7, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        # 核心注意力块
        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        # ----------------- 开启深度监督：辅助预测头 -----------------
        self.ds_head_c4 = nn.Conv2d(c4_in_channels, self.num_classes, kernel_size=1)
        self.ds_head_c3 = nn.Conv2d(c3_in_channels, self.num_classes, kernel_size=1)
        self.ds_head_c2 = nn.Conv2d(c2_in_channels, self.num_classes, kernel_size=1)
        # --------------------------------------------------------

        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)

        # Stage 4
        c_key = self.cat_key1([c4, c3, c2, c1]).flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4.flatten(2).transpose(1, 2), c_key, h4, w4, h4, w4)
        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)

        # Stage 3
        c_key = self.cat_key2([_c4, c3, c2, c1]).flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3.flatten(2).transpose(1, 2), c_key, h4, w4, h3, w3)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)

        # Stage 2
        c_key = self.cat_key3([_c4, _c3, c2, c1]).flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2.flatten(2).transpose(1, 2), c_key, h4, w4, h2, w2)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)

        # Stage 1
        c_key = self.cat_key4([_c4, _c3, _c2, c1]).flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1.flatten(2).transpose(1, 2), c_key, h4, w4, h1, w1)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

        # --- 深度监督分支 ---
        if self.training:
            out_ds4 = resize(self.ds_head_c4(_c4), size=target_size, mode='bilinear', align_corners=self.align_corners)
            out_ds3 = resize(self.ds_head_c3(_c3), size=target_size, mode='bilinear', align_corners=self.align_corners)
            out_ds2 = resize(self.ds_head_c2(_c2), size=target_size, mode='bilinear', align_corners=self.align_corners)

        # --- 主预测分支 ---
        _c4_up = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3_up = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2_up = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)

        _fuse = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1], dim=1))
        x_main = self.linear_pred(self.dropout(_fuse))
        x_main = resize(x_main, size=target_size, mode='bilinear', align_corners=self.align_corners)

        if self.training:
            return [x_main, out_ds4, out_ds3, out_ds2]
        else:
            return x_main

    # ======================== 核心修复：重写损失计算方法 ========================
    def loss_by_feat(self, seg_logits, batch_data_samples):
        """重写此方法以支持 List 类型的 seg_logits"""
        loss = dict()
        
        # 如果是列表（训练模式），则循环计算每个分支的损失
        if isinstance(seg_logits, list):
            # seg_logits[0] 是主预测，[1:] 是辅助预测
            # 我们将主预测的损失记录为 'loss_ce'，辅助分支记录为 'loss_ds_0', 'loss_ds_1' 等
            main_loss = super().loss_by_feat(seg_logits[0], batch_data_samples)
            loss.update(main_loss)

            for i, logits in enumerate(seg_logits[1:]):
                ds_loss = super().loss_by_feat(logits, batch_data_samples)
                # 为避免 key 冲突，重命名辅助分支的损失项
                for key, val in ds_loss.items():
                    loss[f'{key}_ds_{i}'] = val * 0.4 # 辅助头通常给 0.4 的权重
        else:
            # 推理模式下，直接调用基类方法
            loss.update(super().loss_by_feat(seg_logits, batch_data_samples))
            
        return loss



# ======================== 更新后的 APFormerHead3 ========================
@MODELS.register_module()
class APFormerHead_SpinePro3(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead_SpinePro3, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        # 【重点修改】这里使用集成了 Delta Rule 的 RPDA_Block 替换原来的 Block
        self.attn_c4 = RPDA_Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = RPDA_Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = RPDA_Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = RPDA_Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c4_t = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4_t, c_key, h4, w4, h4, w4)

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        c_key = self.cat_key2([_c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c3_t = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_t, c_key, h3, w3, h4, w4)  # 注意保持原先的宽高传参顺序

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3([_c4, _c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c2_t = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_t, c_key, h2, w2, h4, w4)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4([_c4, _c3, _c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c1_t = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1_t, c_key, h1, w1, h4, w4)

        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
        x = resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
        return x
class BoundaryFocalDiceLoss(nn.Module):
    def __init__(self, smooth=1e-8, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def dice_loss(self, pred, target):
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        
        target_tmp = target.clone()
        target_tmp[target == self.ignore_index] = 0
        if torch.any(target_tmp >= num_classes):
            target_tmp[target_tmp >= num_classes] = 0

        target_one_hot = F.one_hot(target_tmp, num_classes=num_classes).permute(0, 3, 1, 2).float()
        mask = (target != self.ignore_index).float().unsqueeze(1)
        pred_soft = pred_soft * mask
        target_one_hot = target_one_hot * mask

        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def boundary_loss(self, pred, target):
        num_classes = pred.shape[1]
        # 针对前景 index 1
        pred_soft = F.softmax(pred, dim=1)[:, 1:2, :, :]
        
        target_tmp = target.clone()
        target_tmp[target == self.ignore_index] = 0
        target_tmp[target_tmp >= num_classes] = 0
        
        target_one_hot = F.one_hot(target_tmp, num_classes=num_classes).permute(0, 3, 1, 2)[:, 1:2, :, :].float()
        mask = (target != self.ignore_index).float().unsqueeze(1)
        pred_soft = pred_soft * mask
        target_one_hot = target_one_hot * mask

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

        t_edge = torch.abs(F.conv2d(target_one_hot, sobel_x, padding=1)) + torch.abs(F.conv2d(target_one_hot, sobel_y, padding=1))
        p_edge = torch.abs(F.conv2d(pred_soft, sobel_x, padding=1)) + torch.abs(F.conv2d(pred_soft, sobel_y, padding=1))

        t_edge = (t_edge > 0.1).float()
        p_edge = (p_edge > 0.1).float()
        return F.mse_loss(p_edge, t_edge)

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal = alpha * (1 - p_t) ** gamma * ce_loss
        return focal.mean()

    def forward(self, pred, target):
        dl = self.dice_loss(pred, target)
        bl = self.boundary_loss(pred, target)
        fl = self.focal_loss(pred, target)
        return 0.3 * dl + 0.4 * bl + 0.3 * fl


# ====================== 2. 修改后的原始 Head ======================
@MODELS.register_module()
class APFormerHead_SpinePro1(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead_SpinePro1, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # 【新增】实例化三合一损失函数
        self.bfd_loss = BoundaryFocalDiceLoss(ignore_index=self.ignore_index)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c4_t = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4_t, c_key, h4, w4, h4, w4)

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        c_key = self.cat_key2([_c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c3_t = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_t, c_key, h4, w4, h3, w3)

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3([_c4, _c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c2_t = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_t, c_key, h4, w4, h2, w2)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4([_c4, _c3, _c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c1_t = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1_t, c_key, h4, w4, h1, w1)

        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
        x = resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
        return x

    # 【新增】重写损失计算函数，替换默认的 CrossEntropy
    def loss_by_feat(self, seg_logits, batch_data_samples):
        """从特征计算损失。"""
        loss = dict()
        # 1. 提取标签并对齐设备
        seg_label = torch.stack([d.gt_sem_seg.data for d in batch_data_samples], dim=0).squeeze(1).long()
        
        # 2. 计算三合一损失
        # 注意：seg_logits 是 forward 返回的结果，即预测图
        loss['loss_bfd'] = self.bfd_loss(seg_logits, seg_label)

        # 3. 计算辅助精度指标（可选，方便观察训练情况）
        with torch.no_grad():
            pred = seg_logits.argmax(1)
            mask = seg_label != self.ignore_index
            acc = ((pred == seg_label) & mask).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0, device=pred.device)
            loss['acc_seg'] = acc

        return loss
@MODELS.register_module()

class APFormerHead_SpinePro4(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead_SpinePro4, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        # 初始化 CA 模块
        self.ca1 = CoordinateAttention(c1_in_channels, c1_in_channels)
        self.ca2 = CoordinateAttention(c2_in_channels, c2_in_channels)
        self.ca3 = CoordinateAttention(c3_in_channels, c3_in_channels)
        self.ca4 = CoordinateAttention(c4_in_channels, c4_in_channels)

        # 初始化 RPDA 注意力块
        self.attn_c4 = RPDA_Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = RPDA_Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = RPDA_Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = RPDA_Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)
        
        # 【新增】初始化多尺度残差细化模块 (针对每个尺度)
        self.refine_c4 = MultiScaleRefine(c4_in_channels)
        self.refine_c3 = MultiScaleRefine(c3_in_channels)
        self.refine_c2 = MultiScaleRefine(c2_in_channels)
        self.refine_c1 = MultiScaleRefine(c1_in_channels)

        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        
        # 1. 坐标注意力增强主干特征
        c1 = self.ca1(c1)
        c2 = self.ca2(c2)
        c3 = self.ca3(c3)
        c4 = self.ca4(c4)

        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # Stage 4
        c_key = self.cat_key1([c4, c3, c2, c1]).flatten(2).transpose(1, 2)
        c4_t = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4_t, c_key, h4, w4, h4, w4)
        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        _c4 = self.refine_c4(_c4) # 【新增】交叉注意力后进行空间洗练

        # Stage 3
        c_key = self.cat_key2([_c4, c3, c2, c1]).flatten(2).transpose(1, 2)
        c3_t = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_t, c_key, h3, w3, h4, w4)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        _c3 = self.refine_c3(_c3) # 【新增】

        # Stage 2
        c_key = self.cat_key3([_c4, _c3, c2, c1]).flatten(2).transpose(1, 2)
        c2_t = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_t, c_key, h2, w2, h4, w4)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        _c2 = self.refine_c2(_c2) # 【新增】

        # Stage 1
        c_key = self.cat_key4([_c4, _c3, _c2, c1]).flatten(2).transpose(1, 2)
        c1_t = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1_t, c_key, h1, w1, h4, w4)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)
        _c1 = self.refine_c1(_c1) # 【新增】

        # 融合与上采样
        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
        x = resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
        return x

@MODELS.register_module()
class APFormerHead(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=8, mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=5, mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=2, mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=1, mlp_ratio=4, drop_path=0.1, pool_ratio=1)
        
        pool_ratio=[1, 2, 4, 8]
        self.cat_key = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=(c1_in_channels + c2_in_channels + c3_in_channels + c4_in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key([c4, c3, c2, c1])
        c1_token = c1.flatten(2).transpose(1, 2)
        c2_token = c2.flatten(2).transpose(1, 2)
        c3_token = c3.flatten(2).transpose(1, 2)
        c4_token = c4.flatten(2).transpose(1, 2)
        c_key_token = c_key.flatten(2).transpose(1, 2)

        _c4 = self.attn_c4(c4_token, c_key_token, h4, w4, h4, w4)
        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)

        _c3 = self.attn_c3(c3_token, c_key_token, h4, w4, h3, w3)
        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)

        _c2 = self.attn_c2(c2_token, c_key_token, h4, w4, h2, w2)
        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        _c2 = resize(_c2, size=(h1, w1), mode='bilinear', align_corners=False)

        _c1 = self.attn_c1(c1_token, c_key_token, h4, w4, h1, w1)
        _c1 = _c1.permute(0, 2, 1).reshape(n, -1, h1, w1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        # 🔧 关键修正：对齐输出尺寸
        if data_samples is not None:
            # 优先缩放到原始图像尺寸，防止评估时 shape 不匹配
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
            
        x = resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
        return x

@MODELS.register_module()
class APFormerHead2(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead2, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c4_t = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4_t, c_key, h4, w4, h4, w4)

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        c_key = self.cat_key2([_c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c3_t = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_t, c_key, h4, w4, h3, w3)

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3([_c4, _c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c2_t = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_t, c_key, h4, w4, h2, w2)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4([_c4, _c3, _c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c1_t = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1_t, c_key, h4, w4, h1, w1)

        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
        x = resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
        return x

@MODELS.register_module()
class APFormerHead2_rebuttal(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead2_rebuttal, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=c4_in_channels + c3_in_channels, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=c4_in_channels + c3_in_channels + c2_in_channels, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=c3_in_channels + c2_in_channels + c1_in_channels, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=c2_in_channels + c1_in_channels, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKey(pool_ratio=[1, 2], dim=[c4_in_channels, c3_in_channels])
        self.cat_key2 = CatKey(pool_ratio=[1, 2, 4], dim=[c4_in_channels, c3_in_channels, c2_in_channels])
        self.cat_key3 = CatKey(pool_ratio=[2, 4, 8], dim=[c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=[4, 8], dim=[c2_in_channels, c1_in_channels])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key1([c4, c3])
        c_key = c_key.flatten(2).transpose(1, 2)
        c4_t = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4_t, c_key, h4, w4, h4, w4)

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        c_key = self.cat_key2([_c4, c3, c2])
        c_key = c_key.flatten(2).transpose(1, 2)
        c3_t = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_t, c_key, h4, w4, h3, w3)

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3([_c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c2_t = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_t, c_key, h4, w4, h2, w2)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4([_c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2)
        c1_t = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1_t, c_key, h4, w4, h1, w1)

        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
        x = resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
        return x

@MODELS.register_module()
class APFormerHeadMulti(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHeadMulti, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        num_heads = decoder_params['num_heads']
        pool_ratio = decoder_params['pool_ratio']
        num_Multi = decoder_params['num_multi']

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels+c3_in_channels*num_Multi, num_heads=num_heads[0], mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels+c3_in_channels*num_Multi, num_heads=num_heads[1], mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels+c3_in_channels*num_Multi, num_heads=num_heads[2], mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels+c3_in_channels*num_Multi, num_heads=num_heads[3], mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKeyMulti(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels], num_feat = num_Multi)
        self.cat_key2 = CatKeyMulti(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels], num_feat = num_Multi)
        self.cat_key3 = CatKeyMulti(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels], num_feat = num_Multi)
        self.cat_key4 = CatKeyMulti(pool_ratio=pool_ratio, dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels], num_feat = num_Multi)

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, data_samples=None):
        if isinstance(inputs, (tuple, list)):
            x_feat = self._transform_inputs(inputs[0])
        else:
            x_feat = self._transform_inputs(inputs)
            
        c1, c2, c3, c4 = x_feat
        xMulti = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None
        
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        if xMulti is not None:
            c_key = self.cat_key1([c4, c3, c2, c1], xMulti)
            c_key = c_key.flatten(2).transpose(1, 2)
            c4_t = c4.flatten(2).transpose(1, 2)
            _c4 = self.attn_c4(c4_t, c_key, h4, w4, h4, w4)

            _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
            c_key = self.cat_key2([_c4, c3, c2, c1], xMulti)
            c_key = c_key.flatten(2).transpose(1, 2)
            c3_t = c3.flatten(2).transpose(1, 2)
            _c3 = self.attn_c3(c3_t, c_key, h4, w4, h3, w3)

            _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
            c_key = self.cat_key3([_c4, _c3, c2, c1], xMulti)
            c_key = c_key.flatten(2).transpose(1, 2)
            c2_t = c2.flatten(2).transpose(1, 2)
            _c2 = self.attn_c2(c2_t, c_key, h4, w4, h2, w2)

            _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
            c_key = self.cat_key4([_c4, _c3, _c2, c1], xMulti)
            c_key = c_key.flatten(2).transpose(1, 2)
            c1_t = c1.flatten(2).transpose(1, 2)
            _c1 = self.attn_c1(c1_t, c_key, h4, w4, h1, w1)

            _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
            _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
            _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
            _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
            x = self.dropout(_c)
            x = self.linear_pred(x)
        else:
            x = torch.zeros((n, self.num_classes, h1, w1), device=c4.device)

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
        x = resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
        return x

class CatKey_single(nn.Module):
    def __init__(self, pool_ratio=1, dim=1):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.pool_list = nn.AvgPool2d(self.pool_ratio, self.pool_ratio, ceil_mode=True)

    def forward(self, x):
        return self.sr_list(self.pool_list(x))

@MODELS.register_module()
class APFormerHeadSingle(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHeadSingle, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        tot_channels = sum(self.in_channels)

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=c4_in_channels, num_heads=8, mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=c4_in_channels, num_heads=5, mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=c3_in_channels, num_heads=2, mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=c2_in_channels, num_heads=1, mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKey_single(pool_ratio=1, dim=c4_in_channels)
        self.cat_key2 = CatKey_single(pool_ratio=1, dim=c4_in_channels)
        self.cat_key3 = CatKey_single(pool_ratio=2, dim=c3_in_channels)
        self.cat_key4 = CatKey_single(pool_ratio=4, dim=c2_in_channels)

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key1(c4)
        c_key = c_key.flatten(2).transpose(1, 2)
        c4_t = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4_t, c_key, h4, w4, h4, w4)

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        c_key = self.cat_key2(_c4)
        c_key = _c4.flatten(2).transpose(1, 2)
        c3_t = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_t, c_key, h4, w4, h3, w3)

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3(_c3)
        c_key = c_key.flatten(2).transpose(1, 2)
        c2_t = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_t, c_key, h4, w4, h2, w2)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4(_c2)
        c_key = c_key.flatten(2).transpose(1, 2)
        c1_t = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1_t, c_key, h4, w4, h1, w1)

        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        if data_samples is not None:
            target_size = data_samples[0].metainfo['img_shape'][:2]
        else:
            target_size = (h1 * 4, w1 * 4)
        x = resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
        return x