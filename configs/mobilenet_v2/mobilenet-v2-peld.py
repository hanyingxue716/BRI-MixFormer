# ====================== 1. 基础继承（只依赖你环境里有的文件） ======================
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_spine.py'
]

# ====================== 2. 模型：MobileNetV2 + DeepLabV3（老版本兼容） ======================
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
    # MobileNetV2 主干
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='mmcls://mobilenet_v2')
    ),
    # 🔥 老版本 MMSeg 自带的 ASPPHead = DeepLabV3 核心
    decode_head=dict(
        type='ASPPHead',
        in_channels=320,
        in_index=3,
        channels=512,
        dilations=(1, 6, 12, 18),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    # 辅助头
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=96,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ====================== 3. 数据集元信息 ======================
peld_metainfo = dict(
    classes=('background', 'peld_target'),
    palette=[[0, 0, 0], [255, 255, 255]]
)

# ====================== 4. 你的医学影像增强（完全一样） ======================
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

# ====================== 5. 你的数据集路径 ======================
dataset_type = 'BaseSegDataset'
data_root = '/root/autodl-tmp/u-mixformer-main/data/PELD/'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
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

# ====================== 6. 你的评估指标 ======================
val_evaluator = dict(
    type='SampleWiseIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'mFscore'],
    ignore_index=255,
    nan_to_num=0,
)
test_evaluator = val_evaluator

# ====================== 7. 你的训练策略 ======================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=25000, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True,
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
    dict(type='EarlyStoppingHook', monitor='mDice', rule='greater', min_delta=0.001, patience=200, strict=False)
]

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')