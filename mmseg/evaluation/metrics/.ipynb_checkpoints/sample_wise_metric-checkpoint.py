import torch
import numpy as np
import torch.nn.functional as F
from mmseg.registry import METRICS
from mmseg.evaluation.metrics.iou_metric import IoUMetric


@METRICS.register_module()
class SampleWiseIoUMetric(IoUMetric):
    """
    Sample-wise mean IoU/Dice for medical image segmentation.
    Computes metrics per image, then averages across the dataset.
    """
    def __init__(self, 
                 iou_metrics=['mIoU', 'mDice'],
                 ignore_index=255,
                 nan_to_num=0,
                 **kwargs):
        super().__init__(
            iou_metrics=iou_metrics,
            ignore_index=ignore_index,
            nan_to_num=nan_to_num,
            **kwargs
        )
        self.results = []

    def process(self, data_batch: dict, data_samples: list) -> None:
        for data_sample in data_samples:
            pred_obj = getattr(data_sample, 'pred_sem_seg', None) or data_sample.get('pred_sem_seg', None)
            gt_obj = getattr(data_sample, 'gt_sem_seg', None) or data_sample.get('gt_sem_seg', None)

            if pred_obj is None:
                pred_obj = getattr(data_sample, 'seg_logits', None) or data_sample.get('seg_logits', None)
            if pred_obj is None or gt_obj is None:
                continue

            pred = self._unwrap_to_tensor(pred_obj)
            gt = self._unwrap_to_tensor(gt_obj)
            if pred is None or gt is None:
                continue

            if pred.ndim == 4:
                pred = pred.squeeze(0)
            if gt.ndim == 4:
                gt = gt.squeeze(0)

            metrics = self._compute_sample_metrics(pred, gt)
            self.results.append(metrics)

    def _unwrap_to_tensor(self, item):
        if item is None:
            return None
        if hasattr(item, 'data'):
            item = item.data
        if isinstance(item, dict):
            item = item.get('data', item.get('sem_seg', item))
        if not isinstance(item, torch.Tensor):
            try:
                item = torch.as_tensor(item, dtype=torch.float32)
            except Exception:
                return None
        return item

    def _compute_sample_metrics(self, pred, gt):
        # ----------------------- 修复：转float，再resize -----------------------
        if pred.dtype == torch.long:
            pred = pred.float()

        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(0),
                size=gt.shape[-2:],
                mode='nearest',
                align_corners=None
            ).squeeze(0)
        # ---------------------------------------------------------------------

        if pred.dim() == 3 and pred.shape[0] > 1:
            pred = torch.argmax(pred, dim=0)
        elif pred.dim() == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        gt = gt.long()
        if gt.dim() == 3:
            gt = gt.squeeze(0)

        valid_mask = (gt != self.ignore_index)
        if valid_mask.sum() == 0:
            return {'dice': 0.0, 'iou': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

        pred_m = pred[valid_mask]
        gt_m = gt[valid_mask]

        tp = torch.logical_and(pred_m == 1, gt_m == 1).sum().item()
        fp = torch.logical_and(pred_m == 1, gt_m == 0).sum().item()
        fn = torch.logical_and(pred_m == 0, gt_m == 1).sum().item()

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)

        return {'dice': float(dice), 'iou': float(iou), 'tp': tp, 'fp': fp, 'fn': fn}

    def compute_metrics(self, results: list) -> dict:
        if not results:
            return {'mDice': 0.0, 'mIoU': 0.0, 'mPrecision': 0.0, 'mRecall': 0.0, 'mFscore': 0.0}

        dices = [r['dice'] for r in results]
        ious = [r['iou'] for r in results]
        tps = [r['tp'] for r in results]
        fps = [r['fp'] for r in results]
        fns = [r['fn'] for r in results]

        prec = [tp / (tp + fp) if (tp + fp) > 0 else 0.0 for tp, fp in zip(tps, fps)]
        rec = [tp / (tp + fn) if (tp + fn) > 0 else 0.0 for tp, fn in zip(tps, fns)]
        f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(prec, rec)]

        return {
            'mDice': float(np.mean(dices)),
            'mIoU': float(np.mean(ious)),
            'mPrecision': float(np.mean(prec)),
            'mRecall': float(np.mean(rec)),
            'mFscore': float(np.mean(f1)),
            'mDice_std': float(np.std(dices)),
        }