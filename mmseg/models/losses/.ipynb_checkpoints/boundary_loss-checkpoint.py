import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

@MODELS.register_module()
class BoundaryLoss(nn.Module):
    """
    通过形态学膨胀(Max Pooling)提取边界，计算边界区域的损失。
    专用于压低 HD95 和提升医学图像的边缘贴合度。
    """
    def __init__(self, loss_weight=1.0, pool_size=5, loss_name='loss_boundary'):
        super(BoundaryLoss, self).__init__()
        self.loss_weight = loss_weight
        self.pool_size = pool_size
        self._loss_name = loss_name

    def forward(self, cls_score, label, **kwargs):
        """
        cls_score: (B, C, H, W) - 模型的预测输出 (未经过 softmax/sigmoid)
        label: (B, H, W) - 真实的 Mask 标签
        """
        # 1. 获取前景类的概率图 (针对多分类，假设索引 1 是 spine)
        if cls_score.shape[1] > 1:
            pred = F.softmax(cls_score, dim=1)[:, 1, :, :] 
        else:
            pred = torch.sigmoid(cls_score).squeeze(1)
            
        target = label.float()
        if target.dim() == 4:
            target = target.squeeze(1)
            
        pred_unsqueeze = pred.unsqueeze(1)    # (B, 1, H, W)
        target_unsqueeze = target.unsqueeze(1) # (B, 1, H, W)
        
        # 2. 形态学膨胀提取边界 (Dilation - Original = Boundary)
        # pool_size 控制了边界的粗细程度
        pred_pool = F.max_pool2d(
            pred_unsqueeze, 
            kernel_size=self.pool_size, 
            stride=1, 
            padding=self.pool_size // 2
        )
        target_pool = F.max_pool2d(
            target_unsqueeze, 
            kernel_size=self.pool_size, 
            stride=1, 
            padding=self.pool_size // 2
        )
        
        # 得到软边界图
        pred_boundary = pred_pool - pred_unsqueeze
        target_boundary = target_pool - target_unsqueeze
        
        # 3. 计算边界区域的 Dice 损失
        intersect = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum()
        
        boundary_dice = (2. * intersect + 1e-5) / (union + 1e-5)
        loss = 1 - boundary_dice
        
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name