# ====================== 1. 核心修复：强制点名注册官方模块 ======================
custom_imports = dict(
    imports=['mmseg', 'mmseg.models.decode_heads.feedformer_head'], 
    allow_failed_imports=False)

# 基础运行时环境依然继承
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_spine.py'
]

# ====================== 修复 CuBLAS 报错（必删） ======================
randomness = dict(seed=0)
find_unused_parameters = True
crop_size = (512, 512)

# ====================== 息肉数据集元信息 ======================
xirou_metainfo = dict(
    classes=('background', 'polyp'),
    palette=[[0, 0, 0], [255, 255, 255]]
)

# ====================== 2. 模型配置 (MiT-B0 + FeedFormer) ======================
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size_divisor=32,      # 🔥 修复 AssertionError
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_val=0,
        seg_pad_val=255),
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth')),
    decode_head=dict(
        type='FeedFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),  # 🔥 官方标配
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ====================== 3. 数据增强 ✅ 完全保留你原来的不变 ======================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.70),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', degree=(-15, 15), pad_val=0, seg_pad_val=255, prob=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=30, contrast_range=(0.8, 1.2)),
    dict(type='Pad', size=crop_size, pad_val=0),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ====================== 4. 息肉数据集路径 ======================
dataset_type = 'BaseSegDataset'
data_root = '/root/autodl-tmp/u-mixformer-main/data/xirou/'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=xirou_metainfo,
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        data_prefix=dict(img_path='train/images', seg_map_path='train/labels'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=xirou_metainfo,
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        data_prefix=dict(img_path='test/Kvasir/images', seg_map_path='test/Kvasir/labels'),
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='SampleWiseIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'mFscore'],
    ignore_index=255,
    nan_to_num=0,
)
test_evaluator = val_evaluator

# ====================== 5. 训练策略：30000 迭代 + 500 轮验证一次 ======================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=30000, by_epoch=False)
]

# ====================== 6. Hooks ======================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=1000, 
        save_best='mDice', rule='greater', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mDice',
        rule='greater',
        min_delta=0.001,
        patience=300,
        strict=False)
]

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')