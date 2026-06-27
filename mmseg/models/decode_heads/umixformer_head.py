# ---------------------------------------------------------------
# Copyright (c) 2021, Nota AI GmbH. All rights reserved.
# Enhanced with Dynamic Gating, Deep Supervision & BFD Loss
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
import math
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from mmseg.models.builder import HEADS  # 

# ====================== 1. 基础模块定义 ======================
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

# ====================== 3. 注意力与 Block ======================
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

class CatKey_single(nn.Module):
    def __init__(self, pool_ratio=1, dim=1):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.pool_list = nn.AvgPool2d(self.pool_ratio, self.pool_ratio, ceil_mode=True)

    def forward(self, x):
        return self.sr_list(self.pool_list(x))



class AuxHead(nn.Module):
    def __init__(self, in_c, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c//2, 3, padding=1),
            nn.BatchNorm2d(in_c//2),
            nn.GELU(),
            nn.Conv2d(in_c//2, num_classes, 1)
        )
    def forward(self, x):
        return self.conv(x)

# ======================== 原版全部保留（解决导入错误） ========================
@MODELS.register_module()
class APFormerHead(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        self.feature_strides = feature_strides
        c1, c2, c3, c4 = self.in_channels
        tot = sum(self.in_channels)
        emb_dim = kwargs['decoder_params']['embed_dim']
        self.attn_c4 = Block(dim1=c4, dim2=tot, num_heads=8, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3, dim2=tot, num_heads=5, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2, dim2=tot, num_heads=2, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1, dim2=tot, num_heads=1, pool_ratio=1)
        self.cat_key = CatKey(pool_ratio=[1, 2, 4, 8], dim=[c4, c3, c2, c1])
        self.linear_fuse = ConvModule(tot, emb_dim, 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.linear_pred = nn.Conv2d(emb_dim, self.num_classes, 1)

    def forward(self, inputs, data_samples=None):
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        n, _, h4, w4 = c4.shape
        h1, w1 = c1.shape[2:]
        c_k = self.cat_key([c4, c3, c2, c1]).flatten(2).transpose(1, 2)
        _c4 = resize(self.attn_c4(c4.flatten(2).transpose(1, 2), c_k, h4, w4, h4, w4).permute(0,2,1).reshape(n,-1,h4,w4), size=(h1,w1))
        _c3 = resize(self.attn_c3(c3.flatten(2).transpose(1, 2), c_k, h4, w4, c3.shape[2], c3.shape[3]).permute(0,2,1).reshape(n,-1,c3.shape[2],c3.shape[3]), size=(h1,w1))
        _c2 = resize(self.attn_c2(c2.flatten(2).transpose(1, 2), c_k, h4, w4, c2.shape[2], c2.shape[3]).permute(0,2,1).reshape(n,-1,c2.shape[2],c2.shape[3]), size=(h1,w1))
        _c1 = self.attn_c1(c1.flatten(2).transpose(1, 2), c_k, h4, w4, h1, w1).permute(0,2,1).reshape(n,-1,h1,w1)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.linear_pred(self.dropout(_c))
        target_size = data_samples[0].metainfo['img_shape'][:2] if data_samples else (h1*4, w1*4)
        return resize(x, size=target_size, mode='bilinear', align_corners=self.align_corners)

@MODELS.register_module()
class APFormerHead2(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead2, self).__init__(input_transform='multiple_select', **kwargs)
        c1, c2, c3, c4 = self.in_channels
        tot = sum(self.in_channels)
        params = kwargs['decoder_params']
        self.attn_c4 = Block(dim1=c4, dim2=tot, num_heads=params['num_heads'][0], pool_ratio=8)
        self.attn_c3 = Block(dim1=c3, dim2=tot, num_heads=params['num_heads'][1], pool_ratio=4)
        self.attn_c2 = Block(dim1=c2, dim2=tot, num_heads=params['num_heads'][2], pool_ratio=2)
        self.attn_c1 = Block(dim1=c1, dim2=tot, num_heads=params['num_heads'][3], pool_ratio=1)
        self.cat_key1 = CatKey(params['pool_ratio'], [c4,c3,c2,c1])
        self.cat_key2 = CatKey(params['pool_ratio'], [c4,c3,c2,c1])
        self.cat_key3 = CatKey(params['pool_ratio'], [c4,c3,c2,c1])
        self.cat_key4 = CatKey(params['pool_ratio'], [c4,c3,c2,c1])
        self.linear_fuse = ConvModule(tot, params['embed_dim'], 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.linear_pred = nn.Conv2d(params['embed_dim'], self.num_classes, 1)

    def forward(self, inputs, data_samples=None):
        c1, c2, c3, c4 = self._transform_inputs(inputs)
        n, _, h4, w4 = c4.shape
        h1, w1 = c1.shape[2:]
        _c4 = self.attn_c4(c4.flatten(2).transpose(1, 2), self.cat_key1([c4,c3,c2,c1]).flatten(2).transpose(1, 2), h4, w4, h4, w4).permute(0,2,1).reshape(n,-1,h4,w4)
        _c3 = self.attn_c3(c3.flatten(2).transpose(1, 2), self.cat_key2([_c4,c3,c2,c1]).flatten(2).transpose(1, 2), h4, w4, c3.shape[2], c3.shape[3]).permute(0,2,1).reshape(n,-1,c3.shape[2],c3.shape[3])
        _c2 = self.attn_c2(c2.flatten(2).transpose(1, 2), self.cat_key3([_c4,_c3,c2,c1]).flatten(2).transpose(1, 2), h4, w4, c2.shape[2], c2.shape[3]).permute(0,2,1).reshape(n,-1,c2.shape[2],c2.shape[3])
        _c1 = self.attn_c1(c1.flatten(2).transpose(1, 2), self.cat_key4([_c4,_c3,_c2,c1]).flatten(2).transpose(1, 2), h4, w4, h1, w1).permute(0,2,1).reshape(n,-1,h1,w1)
        _f = torch.cat([resize(x, (h1,w1)) for x in [_c4,_c3,_c2]] + [_c1], dim=1)
        x = self.linear_pred(self.dropout(self.linear_fuse(_f)))
        t_sz = data_samples[0].metainfo['img_shape'][:2] if data_samples else (h1*4, w1*4)
        return resize(x, t_sz, mode='bilinear', align_corners=self.align_corners)

@MODELS.register_module()
class APFormerHead2_rebuttal(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHead2_rebuttal, self).__init__(input_transform='multiple_select', **kwargs)
        c1, c2, c3, c4 = self.in_channels
        p = kwargs['decoder_params']
        self.attn_c4 = Block(c4, c4+c3, p['num_heads'][0], pool_ratio=8)
        self.attn_c3 = Block(c3, c4+c3+c2, p['num_heads'][1], pool_ratio=4)
        self.attn_c2 = Block(c2, c3+c2+c1, p['num_heads'][2], pool_ratio=2)
        self.attn_c1 = Block(c1, c2+c1, p['num_heads'][3], pool_ratio=1)
        self.cat_key1 = CatKey([1,2], [c4,c3])
        self.cat_key2 = CatKey([1,2,4], [c4,c3,c2])
        self.cat_key3 = CatKey([2,4,8], [c3,c2,c1])
        self.cat_key4 = CatKey([4,8], [c2,c1])
        self.linear_fuse = ConvModule(sum(self.in_channels), p['embed_dim'], 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.linear_pred = nn.Conv2d(p['embed_dim'], self.num_classes, 1)

    def forward(self, inputs, data_samples=None):
        c1, c2, c3, c4 = self._transform_inputs(inputs)
        n, _, h4, w4 = c4.shape
        h1, w1 = c1.shape[2:]
        _c4 = self.attn_c4(c4.flatten(2).transpose(1, 2), self.cat_key1([c4,c3]).flatten(2).transpose(1, 2), h4, w4, h4, w4).permute(0,2,1).reshape(n,-1,h4,w4)
        _c3 = self.attn_c3(c3.flatten(2).transpose(1, 2), self.cat_key2([_c4,c3,c2]).flatten(2).transpose(1, 2), h4, w4, c3.shape[2], c3.shape[3]).permute(0,2,1).reshape(n,-1,c3.shape[2],c3.shape[3])
        _c2 = self.attn_c2(c2.flatten(2).transpose(1, 2), self.cat_key3([_c3,c2,c1]).flatten(2).transpose(1, 2), h4, w4, c2.shape[2], c2.shape[3]).permute(0,2,1).reshape(n,-1,c2.shape[2],c2.shape[3])
        _c1 = self.attn_c1(c1.flatten(2).transpose(1, 2), self.cat_key4([_c2,c1]).flatten(2).transpose(1, 2), h4, w4, h1, w1).permute(0,2,1).reshape(n,-1,h1,w1)
        _f = torch.cat([resize(x, (h1,w1)) for x in [_c4,_c3,_c2]] + [_c1], dim=1)
        x = self.linear_pred(self.dropout(self.linear_fuse(_f)))
        t_sz = data_samples[0].metainfo['img_shape'][:2] if data_samples else (h1*4, w1*4)
        return resize(x, t_sz, mode='bilinear', align_corners=self.align_corners)

@MODELS.register_module()
class APFormerHeadMulti(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHeadMulti, self).__init__(input_transform='multiple_select', **kwargs)
        c1, c2, c3, c4 = self.in_channels
        tot = sum(self.in_channels)
        p = kwargs['decoder_params']
        num_m = p['num_multi']
        self.attn_c4 = Block(c4, tot+c3*num_m, p['num_heads'][0], pool_ratio=8)
        self.attn_c3 = Block(c3, tot+c3*num_m, p['num_heads'][1], pool_ratio=4)
        self.attn_c2 = Block(c2, tot+c3*num_m, p['num_heads'][2], pool_ratio=2)
        self.attn_c1 = Block(c1, tot+c3*num_m, p['num_heads'][3], pool_ratio=1)
        self.cat_key1 = CatKeyMulti(p['pool_ratio'], [c4,c3,c2,c1], num_feat=num_m)
        self.cat_key2 = CatKeyMulti(p['pool_ratio'], [c4,c3,c2,c1], num_feat=num_m)
        self.cat_key3 = CatKeyMulti(p['pool_ratio'], [c4,c3,c2,c1], num_feat=num_m)
        self.cat_key4 = CatKeyMulti(p['pool_ratio'], [c4,c3,c2,c1], num_feat=num_m)
        self.linear_fuse = ConvModule(tot, p['embed_dim'], 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.linear_pred = nn.Conv2d(p['embed_dim'], self.num_classes, 1)

    def forward(self, inputs, data_samples=None):
        x_feat = self._transform_inputs(inputs[0] if isinstance(inputs, (list, tuple)) else inputs)
        c1, c2, c3, c4 = x_feat
        xMulti = inputs[1] if isinstance(inputs, (list, tuple)) and len(inputs)>1 else None
        n, _, h4, w4 = c4.shape
        h1, w1 = c1.shape[2:]
        if xMulti is not None:
            _c4 = self.attn_c4(c4.flatten(2).transpose(1, 2), self.cat_key1([c4,c3,c2,c1], xMulti).flatten(2).transpose(1, 2), h4, w4, h4, w4).permute(0,2,1).reshape(n,-1,h4,w4)
            _c3 = self.attn_c3(c3.flatten(2).transpose(1, 2), self.cat_key2([_c4,c3,c2,c1], xMulti).flatten(2).transpose(1, 2), h4, w4, c3.shape[2], c3.shape[3]).permute(0,2,1).reshape(n,-1,c3.shape[2],c3.shape[3])
            _c2 = self.attn_c2(c2.flatten(2).transpose(1, 2), self.cat_key3([_c4,_c3,c2,c1], xMulti).flatten(2).transpose(1, 2), h4, w4, c2.shape[2], c2.shape[3]).permute(0,2,1).reshape(n,-1,c2.shape[2],c2.shape[3])
            _c1 = self.attn_c1(c1.flatten(2).transpose(1, 2), self.cat_key4([_c4,_c3,_c2,c1], xMulti).flatten(2).transpose(1, 2), h4, w4, h1, w1).permute(0,2,1).reshape(n,-1,h1,w1)
            _f = torch.cat([resize(x, (h1,w1)) for x in [_c4,_c3,_c2]] + [_c1], dim=1)
            x = self.linear_pred(self.dropout(self.linear_fuse(_f)))
        else:
            x = torch.zeros((n, self.num_classes, h1, w1), device=c4.device)
        t_sz = data_samples[0].metainfo['img_shape'][:2] if data_samples else (h1*4, w1*4)
        return resize(x, t_sz, mode='bilinear', align_corners=self.align_corners)

@MODELS.register_module()
class APFormerHeadSingle(BaseDecodeHead):
    def __init__(self, feature_strides, pool_scales=(1, 2, 3, 6), **kwargs):
        super(APFormerHeadSingle, self).__init__(input_transform='multiple_select', **kwargs)
        c1, c2, c3, c4 = self.in_channels
        tot = sum(self.in_channels)
        emb_dim = kwargs['decoder_params']['embed_dim']
        self.attn_c4 = Block(c4, c4, 8, pool_ratio=8)
        self.attn_c3 = Block(c3, c4, 5, pool_ratio=4)
        self.attn_c2 = Block(c2, c3, 2, pool_ratio=2)
        self.attn_c1 = Block(c1, c2, 1, pool_ratio=1)
        self.cat_key1 = CatKey_single(1, c4)
        self.cat_key2 = CatKey_single(1, c4)
        self.cat_key3 = CatKey_single(2, c3)
        self.cat_key4 = CatKey_single(4, c2)
        self.linear_fuse = ConvModule(tot, emb_dim, 1, norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.linear_pred = nn.Conv2d(emb_dim, self.num_classes, 1)

    def forward(self, inputs, data_samples=None):
        c1, c2, c3, c4 = self._transform_inputs(inputs)
        n, _, h4, w4 = c4.shape
        h1, w1 = c1.shape[2:]
        _c4 = self.attn_c4(c4.flatten(2).transpose(1, 2), self.cat_key1(c4).flatten(2).transpose(1, 2), h4, w4, h4, w4).permute(0,2,1).reshape(n,-1,h4,w4)
        _c3 = self.attn_c3(c3.flatten(2).transpose(1, 2), self.cat_key2(_c4).flatten(2).transpose(1, 2), h4, w4, c3.shape[2], c3.shape[3]).permute(0,2,1).reshape(n,-1,c3.shape[2],c3.shape[3])
        _c2 = self.attn_c2(c2.flatten(2).transpose(1, 2), self.cat_key3(_c3).flatten(2).transpose(1, 2), h4, w4, c2.shape[2], c2.shape[3]).permute(0,2,1).reshape(n,-1,c2.shape[2],c2.shape[3])
        _c1 = self.attn_c1(c1.flatten(2).transpose(1, 2), self.cat_key4(_c2).flatten(2).transpose(1, 2), h4, w4, h1, w1).permute(0,2,1).reshape(n,-1,h1,w1)
        _f = torch.cat([resize(x, (h1,w1)) for x in [_c4,_c3,_c2]] + [_c1], dim=1)
        x = self.linear_pred(self.dropout(self.linear_fuse(_f)))
        t_sz = data_samples[0].metainfo['img_shape'][:2] if data_samples else (h1*4, w1*4)
        return resize(x, t_sz, mode='bilinear', align_corners=self.align_corners)







# =============================================================================
# 基础通用模块 (保持原逻辑，适配MMSeg环境)
# =============================================================================
class DepthWiseConv_PELD_v3(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dw_conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class FeedForward_PELD_v3(nn.Module):
    def __init__(self, in_features: int, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dw_conv = DepthWiseConv_PELD_v3(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x); x = self.dw_conv(x, H, W); x = self.act(x); x = self.dropout(x)
        x = self.fc2(x); x = self.dropout(x)
        return x

class CrossAttention_PELD_v3(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim_q % num_heads == 0
        self.dim_q, self.dim_kv, self.num_heads = dim_q, dim_kv, num_heads
        self.head_dim = dim_q // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or self.head_dim ** -0.5
        self.proj_q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_out = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, Nq, _ = q_feat.shape
        q = self.proj_q(q_feat).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv_feat = self.act(self.norm_kv(kv_feat)) if self.pool_ratio >= 0 else kv_feat
        Bk, Nk, _ = kv_feat.shape
        kv_total = self.proj_kv(kv_feat)
        kv = kv_total.reshape(Bk, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn_score.softmax(dim=-1)
        attn_score = self.attn_drop(attn_score)
        out = (attn_score @ v).transpose(1, 2).reshape(B, Nq, self.dim_q)
        return self.proj_drop(self.proj_out(out))

# =============================================================================
# 🔧 轻量化改进：方案A (Depthwise+Pointwise 替代标准卷积)
# =============================================================================
class NeuronIntegrityMultiScaleJudge_PELD_Integrity_v3(nn.Module):
    def __init__(self, in_dim: int, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim // 4  # 进一步压缩通道
        
        def make_dw_branch(kernel_size: int, padding: int):
            return nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, padding=padding, groups=in_dim, bias=False),
                nn.BatchNorm2d(in_dim), nn.GELU(),
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.GELU()
            )
            
        self.branch_local = make_dw_branch(3, 1)
        self.branch_mid = make_dw_branch(5, 2)
        self.branch_long = make_dw_branch(7, 3)
        self.fusion_conv = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False)
        self.fusion_act = nn.GELU()
        self.mask_pred_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        f_local = self.branch_local(feature_map)
        f_mid = self.branch_mid(feature_map)
        f_long = self.branch_long(feature_map)
        fuse_feat = torch.cat([f_local, f_mid, f_long], dim=1)
        fuse_feat = self.fusion_act(self.fusion_conv(fuse_feat))
        return torch.sigmoid(self.mask_pred_head(fuse_feat))

# =============================================================================
# 迭代细化块 (GIQR Core)
# =============================================================================
class BoundaryIntegrityRefineBlock_PELD_Integrity_v3(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, num_heads=8, mlp_ratio=4., drop_path=0.1, mask_min=0.4):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.norm_q2 = nn.LayerNorm(dim_q)
        self.norm_final = nn.LayerNorm(dim_q)

        self.attn_coarse = CrossAttention_PELD_v3(dim_q, dim_kv, num_heads=num_heads)
        self.integrity_judge = NeuronIntegrityMultiScaleJudge_PELD_Integrity_v3(dim_q)
        self.mask_min = mask_min

        self.boundary_enhance = nn.Sequential(
            nn.Conv2d(dim_q, dim_q, kernel_size=3, padding=1, groups=dim_q),
            nn.BatchNorm2d(dim_q), nn.GELU()
        )
        self.attn_fine = CrossAttention_PELD_v3(dim_q, dim_kv, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ffn = FeedForward_PELD_v3(dim_q, hidden_features=int(dim_q * mlp_ratio))

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor, H: int, W: int):
        B, N, C = q_feat.shape
        coarse_out = self.attn_coarse(self.norm_q1(q_feat), self.norm_kv(kv_feat), H, W)
        coarse_map = coarse_out.transpose(1, 2).reshape(B, C, H, W)
        
        soft_mask = self.integrity_judge(coarse_map)
        soft_mask = torch.clamp(soft_mask, min=self.mask_min, max=1.0)  # 临床安全钳位
        
        filtered_map = coarse_map * soft_mask
        filtered_feat = filtered_map.flatten(2).transpose(1, 2)
        
        enhance_map = self.boundary_enhance(filtered_map)
        enhance_feat = enhance_map.flatten(2).transpose(1, 2)
        q_mid = q_feat + self.drop_path(enhance_feat)
        
        fine_out = self.attn_fine(self.norm_q2(q_mid), self.norm_kv(kv_feat), H, W)
        q_final = q_mid + self.drop_path(fine_out)
        q_final = q_final + self.drop_path(self.ffn(self.norm_final(q_final), H, W))
        return coarse_out, q_final, soft_mask

# =============================================================================
# 特征拼接融合模块
# =============================================================================
class MultiScaleKeyFusion_PELD_v3(nn.Module):
    def __init__(self, pool_ratio_list, channel_list):
        super().__init__()
        self.conv_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        for idx, ratio in enumerate(pool_ratio_list):
            ch = channel_list[idx]
            if ratio > 1:
                self.conv_list.append(nn.Conv2d(ch, ch, kernel_size=1))
                self.pool_list.append(nn.AvgPool2d(ratio, ratio, ceil_mode=True))
            else:
                self.conv_list.append(nn.Identity())
                self.pool_list.append(nn.Identity())

    def forward(self, feat_list):
        fuse_out = []
        for idx, feat in enumerate(feat_list):
            tmp = self.pool_list[idx](feat)
            tmp = self.conv_list[idx](tmp)
            fuse_out.append(tmp)
        return torch.cat(fuse_out, dim=1)

# =============================================================================
# IRB Loss (严格对齐论文 Eq.10~13)
# =============================================================================
class IRBLoss(nn.Module):
    def __init__(self, smooth=1e-8, ignore_index=255, focal_lambda=1.0):
        super().__init__()
        self.smooth = smooth; self.ignore_label = ignore_index; self.focal_lambda = focal_lambda

    def region_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_cls = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        target_clone = target.clone(); target_clone[target == self.ignore_label] = 0
        target_onehot = F.one_hot(target_clone, num_classes=num_cls).permute(0, 3, 1, 2).float()
        valid_mask = (target != self.ignore_label).float().unsqueeze(1)
        
        inter = (pred_soft * target_onehot * valid_mask).sum(dim=(2, 3))
        union = (pred_soft + target_onehot * valid_mask).sum(dim=(2, 3))
        dice_loss = 1 - (2.0 * inter + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()
        
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_label, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** 2 * ce_loss).mean()
        return dice_loss + self.focal_lambda * focal_loss

    def grad_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = F.softmax(pred, dim=1)[:, 1:2]
        B, _, H, W = pred_soft.shape; device = pred.device
        target_clone = target.clone(); target_clone[target == self.ignore_label] = 0
        target_onehot = F.one_hot(target_clone, num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 3, 1, 2)[:, 1:2].float()
        valid_mask = (target != self.ignore_label).float().view(B, 1, H, W)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        grad_pred = torch.abs(F.conv2d(pred_soft, sobel_x, padding=1)) + torch.abs(F.conv2d(pred_soft, sobel_y, padding=1))
        grad_gt = torch.abs(F.conv2d(target_onehot, sobel_x, padding=1)) + torch.abs(F.conv2d(target_onehot, sobel_y, padding=1))
        return F.mse_loss(grad_pred * valid_mask, grad_gt * valid_mask)

    def forward(self, pred_coarse, pred_fine, seg_target, alpha=1.0, beta=1.0):
        return alpha * self.region_loss(pred_coarse, seg_target) + beta * self.grad_loss(pred_fine, seg_target)

# =============================================================================
# 解码器主头 (已修复二分类指标 & 单图平均计算)
# =============================================================================
@MODELS.register_module()
class BoundaryIntegrityIterAttenHead_PELD_Integrity_v3(BaseDecodeHead):
    def __init__(self, **kwargs):
        if "feature_strides" in kwargs: kwargs.pop("feature_strides")
        super().__init__(input_transform="multiple_select", **kwargs)
        c1, c2, c3, c4 = self.in_channels
        decoder_cfg = kwargs["decoder_params"]
        num_heads = decoder_cfg["num_heads"]; pool_ratio = decoder_cfg["pool_ratio"]
        total_channel = sum(self.in_channels)
        self.mask_min = 0.2; self.mask_thresh = 0.5

        self.refine_block_c4 = BoundaryIntegrityRefineBlock_PELD_Integrity_v3(c4, total_channel, num_heads[0], mask_min=self.mask_min)
        self.refine_block_c3 = BoundaryIntegrityRefineBlock_PELD_Integrity_v3(c3, total_channel, num_heads[1], mask_min=self.mask_min)
        self.refine_block_c2 = BoundaryIntegrityRefineBlock_PELD_Integrity_v3(c2, total_channel, num_heads[2], mask_min=self.mask_min)
        self.refine_block_c1 = BoundaryIntegrityRefineBlock_PELD_Integrity_v3(c1, total_channel, num_heads[3], mask_min=self.mask_min)

        self.key_fusion_modules = nn.ModuleList([MultiScaleKeyFusion_PELD_v3(pool_ratio, [c4, c3, c2, c1]) for _ in range(4)])
        self.aux_coarse_c4 = nn.Conv2d(c4, self.num_classes, 1); self.aux_fine_c4 = nn.Conv2d(c4, self.num_classes, 1)
        self.aux_coarse_c3 = nn.Conv2d(c3, self.num_classes, 1); self.aux_fine_c3 = nn.Conv2d(c3, self.num_classes, 1)
        self.aux_coarse_c2 = nn.Conv2d(c2, self.num_classes, 1); self.aux_fine_c2 = nn.Conv2d(c2, self.num_classes, 1)
        self.concat_fuse = ConvModule(total_channel, decoder_cfg["embed_dim"], 1, norm_cfg=dict(type="SyncBN"))
        self.final_pred = nn.Conv2d(decoder_cfg["embed_dim"], self.num_classes, 1)
        self.irb_loss = IRBLoss(ignore_index=self.ignore_index)
        self.wm, self.wi, self.alpha, self.beta = 1.0, [0.5, 0.6, 0.7], 1.0, 0.8

    def forward(self, inputs, data_samples=None):
        feats = self._transform_inputs(inputs)
        c1, c2, c3, c4 = feats
        B, _, H4, W4 = c4.shape; H1, W1 = c1.shape[2:]
        target_size = data_samples[0].metainfo["img_shape"][:2] if data_samples else (H1 * 4, W1 * 4)
        mask_collect = []

        key4 = self.key_fusion_modules[0]([c4, c3, c2, c1]).flatten(2).transpose(1, 2)
        coarse4, fine4, mask4 = self.refine_block_c4(c4.flatten(2).transpose(1, 2), key4, H4, W4)
        mask_collect.append(mask4); coarse4 = coarse4.transpose(1, 2).reshape(B, -1, H4, W4); fine4 = fine4.transpose(1, 2).reshape(B, -1, H4, W4)

        key3 = self.key_fusion_modules[1]([fine4, c3, c2, c1]).flatten(2).transpose(1, 2)
        coarse3, fine3, mask3 = self.refine_block_c3(c3.flatten(2).transpose(1, 2), key3, c3.shape[2], c3.shape[3])
        mask_collect.append(mask3); coarse3 = coarse3.transpose(1, 2).reshape(B, -1, c3.shape[2], c3.shape[3]); fine3 = fine3.transpose(1, 2).reshape(B, -1, c3.shape[2], c3.shape[3])

        key2 = self.key_fusion_modules[2]([fine4, fine3, c2, c1]).flatten(2).transpose(1, 2)
        coarse2, fine2, mask2 = self.refine_block_c2(c2.flatten(2).transpose(1, 2), key2, c2.shape[2], c2.shape[3])
        mask_collect.append(mask2); coarse2 = coarse2.transpose(1, 2).reshape(B, -1, c2.shape[2], c2.shape[3]); fine2 = fine2.transpose(1, 2).reshape(B, -1, c2.shape[2], c2.shape[3])

        key1 = self.key_fusion_modules[3]([fine4, fine3, fine2, c1]).flatten(2).transpose(1, 2)
        coarse1, fine1, _ = self.refine_block_c1(c1.flatten(2).transpose(1, 2), key1, H1, W1)
        coarse1 = coarse1.transpose(1, 2).reshape(B, -1, H1, W1); fine1 = fine1.transpose(1, 2).reshape(B, -1, H1, W1)

        if self.training:
            out_c4_coarse = resize(self.aux_coarse_c4(coarse4), size=target_size, mode="bilinear", align_corners=False)
            out_c4_fine = resize(self.aux_fine_c4(fine4), size=target_size, mode="bilinear", align_corners=False)
            out_c3_coarse = resize(self.aux_coarse_c3(coarse3), size=target_size, mode="bilinear", align_corners=False)
            out_c3_fine = resize(self.aux_fine_c3(fine3), size=target_size, mode="bilinear", align_corners=False)
            out_c2_coarse = resize(self.aux_coarse_c2(coarse2), size=target_size, mode="bilinear", align_corners=False)
            out_c2_fine = resize(self.aux_fine_c2(fine2), size=target_size, mode="bilinear", align_corners=False)
            fuse_feat = torch.cat([resize(fine4, (H1, W1)), resize(fine3, (H1, W1)), resize(fine2, (H1, W1)), fine1], dim=1)
            main_out = resize(self.final_pred(self.concat_fuse(fuse_feat)), size=target_size, mode="bilinear", align_corners=False)
            return [main_out, out_c4_coarse, out_c4_fine, out_c3_coarse, out_c3_fine, out_c2_coarse, out_c2_fine, mask_collect]
        else:
            fuse_feat = torch.cat([resize(fine4, (H1, W1)), resize(fine3, (H1, W1)), resize(fine2, (H1, W1)), fine1], dim=1)
            main_out = resize(self.final_pred(self.concat_fuse(fuse_feat)), size=target_size, mode="bilinear", align_corners=False)
            return main_out

    def loss_by_feat(self, seg_results, batch_data_samples):
        loss_dict = dict()
        seg_label = torch.stack([sample.gt_sem_seg.data for sample in batch_data_samples], dim=0).squeeze(1).long()
        main_pred, c4c, c4f, c3c, c3f, c2c, c2f, mask_list = seg_results

        loss_dict["loss_main_region"] = self.wm * self.irb_loss.region_loss(main_pred, seg_label)
        loss_dict["loss_iter_c4"] = self.wi[0] * self.irb_loss(c4c, c4f, seg_label, self.alpha, self.beta)
        loss_dict["loss_iter_c3"] = self.wi[1] * self.irb_loss(c3c, c3f, seg_label, self.alpha, self.beta)
        loss_dict["loss_iter_c2"] = self.wi[2] * self.irb_loss(c2c, c2f, seg_label, self.alpha, self.beta)

        # ================= 严格单图平均计算 (适配二分类 0/1) =================
        with torch.no_grad():
            B = seg_label.shape[0]
            retain_rates, suppress_rates = [], []
            valid_area = (seg_label != self.ignore_index).float()
            
            for mask in mask_list:
                mask_resized = resize(mask, size=seg_label.shape[-2:], mode="bilinear", align_corners=False)
                mask_resized = torch.clamp(mask_resized, 0.0, 1.0)  # 修复插值越界
                mask_bin = (mask_resized > self.mask_thresh).float().view(B, -1)
                
                neuron = (seg_label == 1).float().view(B, -1) * valid_area.view(B, -1)
                bg = (seg_label == 0).float().view(B, -1) * valid_area.view(B, -1)
                
                # 逐图计算保留率
                r = (mask_bin * neuron).sum(dim=1) / (neuron.sum(dim=1) + 1e-8)
                retain_rates.append(torch.clamp(r, 0.0, 1.0))
                
                # 逐图计算背景/半遮挡抑制率
                s = ((1.0 - mask_bin) * bg).sum(dim=1) / (bg.sum(dim=1) + 1e-8)
                suppress_rates.append(torch.clamp(s, 0.0, 1.0))
                
            loss_dict["valid_neuron_retain_rate"] = torch.stack(retain_rates).mean(dim=0).mean()
            loss_dict["bg_occlusion_suppress_rate"] = torch.stack(suppress_rates).mean(dim=0).mean()
        return loss_dict
















