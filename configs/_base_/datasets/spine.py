# 数据集设置
dataset_type = 'BaseSegDataset' # 脊柱是自定义数据，使用基础类最稳妥
data_root = 'data/spine' # 你的数据根目录

# 训练预处理流程
train_pipeline = [
    dict(type='LoadImageFromFile'), # 读取原图
    dict(type='LoadAnnotations'),    # 读取标注（注意：去掉了 reduce_zero_label）
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5), # 随机翻转
    dict(type='PhotoMetricDistortion'), # 颜色亮度增强
    dict(type='PackSegInputs') # 封装成模型需要的格式
]

# 测试/验证预处理流程
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # 医疗影像建议保持原始标注，不加 reduce_zero_label
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8, # 根据显存调整
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 核心：精准对齐你的目录结构
        data_prefix=dict(
            img_path='img_dir/train', 
            seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/test',
            seg_map_path='ann_dir/test'),
        pipeline=test_pipeline))

# 评估指标：脊柱分割常用 Dice 和 IoU
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = val_evaluator