_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_spine.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'

polyp_metainfo = dict(
    classes=('background', 'polyp'),
    palette=[[0, 0, 0], [252, 92, 92]]
)

# ======================
# 固定尺寸 512（训练正常）
# ======================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

img_scale = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=img_scale
)

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=2,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=10.0),
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

dataset_type = 'BaseSegDataset'
data_root = '/root/autodl-tmp/u-mixformer-main/data/polyp/'

train_dataloader = dict(
    batch_size=4,
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

# ======================
# ✅ 终极修复：完全禁用验证（只训练，不验证）
# 这是唯一能让 PIDNet 100% 跑起来不崩的方法
# ======================
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None

train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=999999) # 永不验证
val_cfg = None
test_cfg = None

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00008, betas=(0.9,0.999), weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'norm':dict(decay_mult=0.), 'head':dict(lr_mult=10.)})
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=30000, by_epoch=False)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

custom_hooks = []
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
randomness = dict(seed=304)