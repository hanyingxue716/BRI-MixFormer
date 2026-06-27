# configs/deeplabv3/deeplabv3_r101-d8_xirou.py

# ====================== 1. 基础配置继承 ======================
_base_ = [
    '../_base_/default_runtime.py'
]

# ResNet-101 预训练权重
checkpoint = 'open-mmlab://resnet101_v1c'

# ====================== 2. 你的息肉数据集元信息 ======================
polyp_metainfo = dict(
    classes=('background', 'polyp'),
    palette=[[0, 0, 0], [252, 92, 92]]
)

# ====================== 3. DeepLabV3 ResNet-101 模型（完全保留）======================
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
    ),
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        multi_grid=(1, 2, 4),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    ),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 6, 12, 18),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type='OHEMPixelSampler', min_kept=100000)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ====================== 4. 你指定的息肉数据增强（1:1 复制）======================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.7),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', degree=(-10, 10), pad_val=0, seg_pad_val=255, prob=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=20, contrast_range=(0.85, 1.15)),
    dict(type='Pad', size=(512, 512), pad_val=0),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ====================== 5. 你的息肉数据集路径 ======================
dataset_type = 'BaseSegDataset'
data_root = '/root/autodl-tmp/u-mixformer-main/data/polyp/'

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=polyp_metainfo,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='TrainDataset/image', seg_map_path='TrainDataset/masks'),
        pipeline=train_pipeline
    )
)

# ====================== 6. 你的 config：无验证、无测试 ======================
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None

# ====================== 7. 你的训练策略 ======================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000000)
val_cfg = None
test_cfg = None

# ====================== 8. 修复后的优化器（删除 _delete_=True）======================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00008, betas=(0.9,0.999), weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={
        'norm': dict(decay_mult=0.),
        'head': dict(lr_mult=10.)
    })
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=20000, by_epoch=False)
]

# ====================== 9. 你的 Hooks：只保存最后一轮，无早停 ======================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=20000,
        max_keep_ckpts=1
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

custom_hooks = []

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)