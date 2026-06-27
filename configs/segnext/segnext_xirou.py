# ====================== 1. 基础配置 ======================
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_spine.py'
]

# ====================== 2. SegNeXt 预训练权重 ======================
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32)
)

# ====================== 3. SegNeXt 模型（2分类 息肉专用）======================
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=2,          # 二分类：背景+息肉
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ====================== 4. 息肉数据集元信息 ======================
polyp_metainfo = dict(
    classes=('background', 'polyp'),
    palette=[[0, 0, 0], [252, 92, 92]]
)

# ====================== 5. 数据增强（完全用你给的配置）======================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ====================== 6. 数据集路径 ======================
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

# ====================== ✅ 验证集（和你完全一样）======================
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=polyp_metainfo,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='TrainDataset/image', seg_map_path='TrainDataset/masks'),
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

# ====================== ✅ 单图平均评估 SampleWiseIoUMetric ======================
val_evaluator = dict(
    type='SampleWiseIoUMetric',
    iou_metrics=['mDice'],
    ignore_index=255,
    nan_to_num=0.0
)
test_evaluator = val_evaluator

# ====================== ✅ 迭代 30000 次 ======================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ====================== 优化器（和你完全一致）======================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00008, betas=(0.9,0.999), weight_decay=0.05),
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
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=30000, by_epoch=False)
]

# ====================== ✅ 保存最佳 mDice 模型 ======================
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
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

custom_hooks = []

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')