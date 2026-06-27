# ====================== 1. 直接继承官方原生 deeplabv3+ 配置 ======================
_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_spine.py'
]

# ====================== 2. 输入尺寸 ======================
crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
)

# ====================== 3. 2 分类 ======================
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
)

# ====================== 4. 息肉数据集元信息 ======================
xirou_metainfo = dict(
    classes=('background', 'polyp'),
    palette=[[0, 0, 0], [255, 255, 255]]
)

# ====================== 5. 你指定的【官方最优数据增强】 ======================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
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

# ====================== 6. 息肉数据集路径 ======================
dataset_type = 'BaseSegDataset'
data_root = '/root/autodl-tmp/u-mixformer-main/data/xirou/'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
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

# ====================== 7. 评估指标 ======================
val_evaluator = dict(
    type='SampleWiseIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'mFscore'],
    ignore_index=255,
    nan_to_num=0,
)
test_evaluator = val_evaluator

# ====================== 8. 训练配置 ======================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=30000, by_epoch=False)
]

# ====================== 9. Hooks ======================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        save_best='mDice',
        rule='greater',
        max_keep_ckpts=1
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