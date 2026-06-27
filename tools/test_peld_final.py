import torch
import torch.nn as nn
import numpy as np
from mmseg.apis import init_model, inference_model
from medpy import metric
import os
import cv2
import time
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

# ====================== 【1. 路径与环境配置】 ======================
config_file = 'configs/umixformer/peld原始版.py'
checkpoint_file = '/root/autodl-tmp/u-mixformer-main/work_dirs/peld损失权重1.50.8/best_mDice_iter_21650.pth'

img_dir = '/root/autodl-tmp/u-mixformer-main/data/PELD/test/image'
ann_dir = '/root/autodl-tmp/u-mixformer-main/data/PELD/test/mask'
save_vis_dir = '/root/autodl-tmp/u-mixformer-main/work_dirs/最好版可视化图'
os.makedirs(save_vis_dir, exist_ok=True)

# 缓存S列表
cache_S = [None]

# hook函数：只捕获refine_block输出第三个return: soft_mask
def capture_S_hook(module, inp, out):
    # out = (coarse_out, q_final, soft_mask)
    _, _, soft_mask = out
    cache_S[0] = soft_mask.detach()

# ====================== 【2. 效率指标计算函数】 ======================
def get_model_complexity(model, input_shape=(1, 3, 512, 512)):
    class WrapModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            if hasattr(self.model, 'backbone'):
                return self.model.backbone(x)
            return self.model(x)

    model.eval()
    device = next(model.parameters()).device
    inputs = torch.randn(*input_shape).to(device)

    wrapped_model = WrapModel(model)
    flops_counter = FlopCountAnalysis(wrapped_model, inputs)
    total_flops = flops_counter.total()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, total_flops

def measure_academic_fps(model, input_shape=(1, 3, 512, 512), warm_up=30, test_num=100):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        for _ in range(warm_up):
            if hasattr(model, 'backbone'):
                model.backbone(dummy_input)
            else:
                model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(test_num):
            if hasattr(model, 'backbone'):
                model.backbone(dummy_input)
            else:
                model(dummy_input)
    torch.cuda.synchronize()
    return test_num / (time.time() - start_time)

# ====================== 【3. 指标函数】 ======================
def calculate_single_metrics(pred, gt):
    def get_binary_metrics(p, g):
        if np.sum(g) == 0:
            return (1.0, 1.0, 1.0, 1.0, 0.0) if np.sum(p) == 0 else (0.0, 0.0, 0.0, 0.0, 50.0)
        
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, np.logical_not(g)).sum()
        fn = np.logical_and(np.logical_not(p), g).sum()
        
        dsc = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        pre = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        try:
            hd = metric.binary.hd95(p, g) if np.sum(p) > 0 else 50.0
        except:
            hd = 50.0
        return dsc, iou, pre, rec, hd

    res_f = get_binary_metrics(pred == 1, gt == 1)
    res_b = get_binary_metrics(pred == 0, gt == 0)
    return res_f, res_b

def calc_retain_suppress(S_np, gt_np, th=0.5, eps=1e-8):
    M = (S_np > th).astype(np.float32)
    G = gt_np.astype(np.float32)
    retain = np.sum(M * G) / (np.sum(G) + eps)
    supp = np.sum((1-M)*(1-G)) / (np.sum(1-G) + eps)
    return retain, supp

# ====================== 【4. 推理主流程】 ======================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_model(config_file, checkpoint_file, device=device)

# 精准绑定类名 BoundaryIntegrityRefineBlock_PELD_Integrity_v3
hook_handle_list = []
for _, submod in model.decode_head.named_modules():
    cls_name = str(type(submod).__name__)
    if "BoundaryIntegrityRefineBlock_PELD_Integrity_v3" == cls_name:
        h = submod.register_forward_hook(capture_S_hook)
        hook_handle_list.append(h)
        # 只绑定第一个（refine_block_c4）即可，多余不绑
        break

print("📊 正在测量模型复杂度与 FPS...")
n_params, n_flops = get_model_complexity(model)
academic_fps = measure_academic_fps(model)

metrics_f_list = []
metrics_b_list = []
retain_list = []
suppress_list = []

img_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
records = []

print(f"🚀 开始原始性能评估 (无后处理, 共 {len(img_list)} 张)...")
for img_name in tqdm(img_list):
    img_path = os.path.join(img_dir, img_name)
    ann_path = os.path.join(ann_dir, img_name)

    cache_S[0] = None
    with torch.no_grad():
        result = inference_model(model, img_path)
    
    pred_mask = result.pred_sem_seg.data.cpu().numpy().squeeze().astype(np.uint8)
    
    gt_raw = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
    if gt_raw is None: continue
    gt_mask = cv2.resize(gt_raw, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    m_f, m_b = calculate_single_metrics(pred_mask, gt_mask)
    metrics_f_list.append(m_f)
    metrics_b_list.append(m_b)

    s_tensor = cache_S[0]
    if s_tensor is not None:
        # s_tensor [B,1,H,W]
        S_np = s_tensor[0,0].cpu().numpy()
        S_np = cv2.resize(S_np, (gt_mask.shape[1], gt_mask.shape[0]))
        ret, sup = calc_retain_suppress(S_np, gt_mask)
        retain_list.append(ret)
        suppress_list.append(sup)
    else:
        retain_list.append(np.nan)
        suppress_list.append(np.nan)

    records.append({'name': img_name, 'dsc_f': m_f[0], 'hd_f': m_f[4], 'pred': pred_mask, 'gt': gt_mask, 'path': img_path})

# ====================== 【5. 汇总输出】 ======================
arr_f = np.array(metrics_f_list)
arr_b = np.array(metrics_b_list)
mean_f = np.mean(arr_f, axis=0)
mean_b = np.mean(arr_b, axis=0)

avg_retain = np.nanmean(np.array(retain_list))
avg_supp = np.nanmean(np.array(suppress_list))

print("\n" + "═" * 85)
print(f"📌 PELD 数据集 - 评估报告 (原始预测版)")
print("═" * 85)
print(f"{'Model Params:':<20} {n_params/1e6:<10.3f} M | {'FLOPs:':<15} {n_flops/1e9:<10.3f} G")
print(f"{'Pure FPS:':<20} {academic_fps:<10.2f} f/s | {'Input:':<15} (3, 512, 512)")
print("-" * 85)
print(f"{'Metric':<15} | {'Foreground (Target)':<20} | {'Background':<15} | {'Mean (Global)':<15}")
print("-" * 85)

labels = ["DSC (Dice)", "IOU", "PRE", "REC", "HD95"]
for i, label in enumerate(labels):
    f_val, b_val = mean_f[i], mean_b[i]
    m_val = (f_val + b_val) / 2
    unit = " px" if label == "HD95" else ""
    print(f"{label:<15} | {f_val:<20.4f}{unit} | {b_val:<15.4f}{unit} | {m_val:<15.4f}{unit}")

print("-" * 85)
print(f"{'有效神经保留率':<15} | {avg_retain:<20.4f} | {'—':<15} | {'—':<15}")
print(f"{'遮挡背景抑制率':<15} | {avg_supp:<20.4f} | {'—':<15} | {'—':<15}")
print("═" * 85)

# ====================== 【可视化】 ======================
sorted_by_dsc = sorted(records, key=lambda x: -x['dsc_f'])
rank_map = {item['name']: idx+1 for idx, item in enumerate(sorted_by_dsc)}

def save_side_by_side(data, folder):
    path = os.path.join(save_vis_dir, folder)
    os.makedirs(path, exist_ok=True)
    
    print(f"🖼️ 正在生成对比图: {folder}...")
    for i, item in enumerate(data):
        img = cv2.imread(item['path'])
        h, w = item['gt'].shape
        img = cv2.resize(img, (w, h))
        
        gt_vis = img.copy()
        gt_vis[item['gt'] == 1] = gt_vis[item['gt'] == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        pred_vis = img.copy()
        pred_vis[item['pred'] == 1] = pred_vis[item['pred'] == 1] * 0.5 + np.array([0, 0, 255]) * 0.5
        tp = np.logical_and(item['pred'] == 1, item['gt'] == 1)
        pred_vis[tp] = img[tp] * 0.5 + np.array([0, 255, 255]) * 0.5
        
        canvas = np.hstack([img, gt_vis, pred_vis])
        rank = rank_map[item['name']]
        info = f"Rank: {rank} | Dice: {item['dsc_f']:.4f} | HD95: {item['hd_f']:.2f} | {item['name']}"
        cv2.putText(canvas, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(canvas, "Original", (w//2-60, h-15), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "GT (Green)", (w + w//2-60, h-15), 0, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, "Pred (Red/Yellow)", (2*w + w//2-80, h-15), 0, 0.8, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(path, item['name']), canvas)

save_side_by_side(records, "all_cases")
save_side_by_side(sorted_by_dsc[:30], "worst_10_cases")
save_side_by_side(sorted_by_dsc[-10:][::-1], "best_10_cases")

print(f"\n✅ 原始评估完成！可视化结果存至: {save_vis_dir}")
print(f"✅ 测试集平均保留率={avg_retain:.4f}, 平均抑制率={avg_supp:.4f}")