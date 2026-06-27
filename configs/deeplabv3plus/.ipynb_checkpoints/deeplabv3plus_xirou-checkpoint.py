# ====================== 1. 直接继承官方原生 deeplabv3+ 配置 ======================


_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',  # ✅ 这是对的
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

# ====================== 3. 2 分类（DeepLabV3+ 结构不动） ======================
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
)

# ====================== 4. 你的息肉数据集元信息 ======================
polyp_metainfo = dict(
    classes=('background', 'polyp'),
    palette=[[0, 0, 0], [252, 92, 92]]
)

# ====================== 5. 你指定的息肉数据增强（1:1 你的配置） ======================
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

# ====================== 6. 你的息肉数据集路径 ======================
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

# ====================== 7. 单图平均评估 SampleWiseIoUMetric 已开启 ======================
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
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='SampleWiseIoUMetric',
    iou_metrics=['mDice'],
    ignore_index=255,
    nan_to_num=0.0
)
test_evaluator = val_evaluator

# ====================== 8. 总迭代 30000 次 ======================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=30000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ====================== 9. 你的优化器 ======================
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
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=30000, by_epoch=False)
]

# ====================== 10. 你的 Hooks ======================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=30000,
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