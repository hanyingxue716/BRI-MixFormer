# 1. 优化器配置
# 医疗影像分割通常建议使用 AdamW，比 SGD 更容易在小数据集上收敛
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper', 
    optimizer=optimizer, 
    clip_grad=dict(max_norm=1.0) # 梯度裁剪，防止医疗数据中的异常梯度导致崩溃
)

# 2. 学习率策略
param_scheduler = [
    # 预热阶段：前 500 步线性增加学习率
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    # 正式训练阶段：余弦退火（比 Poly 在小数据上效果通常更好）
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        begin=500,
        end=20000, # 配合下方的 max_iters
        by_epoch=False)
]

# 3. 训练循环配置
# 针对 1100 张图片，设置 20000 次迭代（约 145 个 Epoch）足够了
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20000, val_interval=1000) # 每 1000 步验证一次
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 4. 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 降低保存频率，每 2000 步保存一个权重，且只保留最好的模型
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=2000, 
        save_best='mDice', # 自动保存验证集 Dice 最高的模型
        max_keep_ckpts=3), # 最多保留 3 个权重文件，节省空间
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))