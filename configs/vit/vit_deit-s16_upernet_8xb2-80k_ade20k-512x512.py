_base_ = './vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py'

model = dict(
    pretrained='checkpoints/segmentation/deit/upernet_deit-s16_512x512_80k_ade20k_20210624_095228-afc93ec2.pth',
    backbone=dict(num_heads=6, embed_dims=384, drop_path_rate=0.1),
    decode_head=dict(num_classes=150, in_channels=[384, 384, 384, 384]),
    neck=None,
    auxiliary_head=dict(num_classes=150, in_channels=384))
