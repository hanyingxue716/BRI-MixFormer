# configs/umixformer/umixformer_mit-b0_peld.py
_base_ = [
    '../_base_/models/segformer_mit-b0-multi.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_spine.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

peld_metainfo = dict(
    classes=('background', 'peld_target'),
    palette=[[0, 0, 0], [255, 255, 255]]
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # 建议 1: 缩小 Resize 范围，保持解剖结构清晰
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.8, 1.2), keep_ratio=True),
    # 建议 2: 调低 cat_max_ratio。保证每个 crop 至少有 30% 是有效视野，而不是只有 5%
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.70), 
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate', degree=(-15, 15), pad_val=0, seg_pad_val=255, prob=0.5),
    # 建议 3: 必须加入光度畸变，这是医学影像的灵魂（模拟出血/冲洗液）
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

# ====================== 绝对兼容、零报错、医学影像专用 ======================

# ====================== 模型 ======================
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        size=(512, 512),
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_val=0,
    ),
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformerMulti',
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
        encoder_params=dict(interval=1),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    ),
    decode_head=dict(
        type='BoundaryIntegrityIterAttenHead_PELD_Integrity_v3',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=2,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        decoder_params=dict(
            embed_dim=256,
            num_heads=[8, 5, 2, 1],
            pool_ratio=[1, 2, 4, 8]
        ),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ====================== 数据集 ======================
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

# ====================== 评估 ======================
val_evaluator = dict(
    type='SampleWiseIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'mFscore'],
    ignore_index=255,
    nan_to_num=0,
)
test_evaluator = val_evaluator

# ====================== 训练配置 ======================
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