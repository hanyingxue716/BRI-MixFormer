# ====================== 1. 基础配置继承 ======================
# 仅保留必要的运行时和基础模型结构，避开 cityscapes 的数据 base
_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/default_runtime.py'
]

# ====================== ✅ B2 官方权重（你给的正确版本） ======================
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

# ====================== 2. 数据集元信息 (PELD 专用) ======================
peld_metainfo = dict(
    classes=('background', 'peld_target'),
    palette=[[0, 0, 0], [255, 255, 255]]
)

# ====================== 3. 数据流水线 (与 U-MixFormer 保持一致) ======================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.70), 
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', degree=(-15, 15), pad_val=0, seg_pad_val=255, prob=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=30, contrast_range=(0.8, 1.2)),
    dict(type='Pad', size=(512, 512), pad_val=0),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ====================== 4. ✅ 模型配置 (SegFormer B2 标准结构) ======================
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(512, 512),
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_val=0,
        seg_pad_val=255,
    ),
    # ====================== ✅ B2 Backbone ======================
    backbone=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],    # B2 标准层数
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    ),
    # ====================== ✅ B2 Head 通道数 ======================
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ====================== 5. 数据加载器 (完全不变) ======================
dataset_type = 'BaseSegDataset'
data_root = '/root/autodl-tmp/u-mixformer-main/data/PELD/'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=peld_metainfo,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='train/image', seg_map_path='train/mask'),
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
        metainfo=peld_metainfo,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='val/image', seg_map_path='val/mask'),
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

# ====================== 6. 评估与训练策略 (完全不变) ======================
val_evaluator = dict(
    type='SampleWiseIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'mFscore'],
    ignore_index=255,
    nan_to_num=0,
)
test_evaluator = val_evaluator

train_cfg = dict(type='IterBasedTrainLoop', max_iters=25000, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ====================== 7. 优化器 (完全不变) ======================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=25000, by_epoch=False)
]

# ====================== 8. Hooks & Visualizer (完全不变) ======================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=100,
        save_best='mDice',
        rule='greater',
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mDice',
        rule='greater',
        min_delta=0.001,
        patience=200,
        strict=False
    )
]

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)