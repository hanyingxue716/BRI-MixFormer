# ====================== 你的最终完美配置 ======================
_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_spine.py'
]

# 基础参数
randomness = dict(seed=0)
find_unused_parameters = True
crop_size = (512, 512)

# 数据预处理
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size_divisor=16,
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
)

# 息肉元信息
xirou_metainfo = dict(
    classes=('background', 'polyp'),
    palette=[[0, 0, 0], [255, 255, 255]]
)

# ====================== 模型核心 ======================
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(with_cp=True),
    decode_head=dict(
        num_classes=1,
        out_channels=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            class_weight=[0.1, 0.9]
        )
    ),
    auxiliary_head=None,
    test_cfg=dict(mode='whole')
)

# ====================== 数据增强 ======================
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

# ✅ 修复：固定尺寸 512,512，满足 UNet 16倍整除要求
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ====================== 已修正：正确路径 ======================
dataset_type = 'BaseSegDataset'
data_root = '/root/autodl-tmp/u-mixformer-main/data/xirou/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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
    num_workers=2,
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

# ====================== 评估 ======================
val_evaluator = dict(
    type='SampleWiseIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'mFscore'],
    ignore_index=255,
    nan_to_num=0,
    class_index=1
)
test_evaluator = val_evaluator

# ====================== 30000轮 + 500轮验证 ======================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=25000, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(norm=dict(decay_mult=0.))
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=1500, end=25000, by_epoch=False)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_best='mDice',
        rule='greater',
        max_keep_ckpts=1
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='mDice', rule='greater', patience=200)
]

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')