#!/usr/bin/env python3
"""
Lab 4 平衡改进版 - LAB 色彩空间 + Color-Boosted Loss + LR Scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import colorsys

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
HIDDEN_CHANNELS = 64
EPOCHS = 5
LEARNING_RATE = 2e-3
VAL_RATIO = 0.1

print(f"🚀 Device: {DEVICE}")
print(f"📊 Training: {EPOCHS} epochs, hidden={HIDDEN_CHANNELS}")

# ============ 色彩空间转换 ============
def rgb2lab(rgb):
    """RGB to LAB (简化版)"""
    # 归一化 RGB
    rgb = rgb.clamp(0, 1)
    # 转线性 RGB
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # XYZ
    M = torch.tensor([[0.4124, 0.3576, 0.1805],
                      [0.2126, 0.7152, 0.0722],
                      [0.0193, 0.1192, 0.9505]], device=rgb.device)
    xyz = torch.matmul(linear.permute(0, 2, 3, 1), M).permute(0, 3, 1, 2)
    # 参考白
    xn, yn, zn = 0.95047, 1.0, 1.08883
    xyz = xyz / torch.tensor([xn, yn, zn], device=rgb.device).view(1, 3, 1, 1)
    # LAB
    def f(t):
        return torch.where(t > 0.008856, t ** (1/3), 7.787 * t + 16/116)
    L = 116 * f(xyz[:, 1:2]) - 16
    a = 500 * (f(xyz[:, 0:1]) - f(xyz[:, 1:2]))
    b = 200 * (f(xyz[:, 1:2]) - f(xyz[:, 2:3]))
    return torch.cat([L, a, b], dim=1)

def lab2rgb(lab):
    """LAB to RGB (简化版)"""
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    # XYZ
    def f_inv(t):
        return torch.where(t > 0.206893, t ** 3, (t - 16/116) / 7.787)
    yn = 1.0
    y = yn * f_inv((L + 16) / 116)
    x = 0.95047 * f_inv((L + 16) / 116 + a / 500)
    z = 1.08883 * f_inv((L + 16) / 116 - b / 200)
    # RGB
    M = torch.tensor([[ 3.2406, -1.5372, -0.4986],
                      [-0.9689,  1.8758,  0.0415],
                      [ 0.0557, -0.2040,  1.0570]], device=lab.device)
    linear = torch.matmul(torch.cat([x, y, z], dim=1).permute(0, 2, 3, 1), M).permute(0, 3, 1, 2)
    rgb = torch.where(linear > 0.0031308, 1.055 * linear ** (1/2.4) - 0.055, 12.92 * linear)
    return rgb.clamp(0, 1)

# ============ 模型 ============
class ImprovedColorCNN_LAB(nn.Module):
    """改进版 CNN - LAB 色彩空间"""
    def __init__(self, hidden=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # Depthwise Separable
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # Dilated
        self.dilated = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # Output: 预测 ab 通道 (2 channels)
        self.out = nn.Conv2d(hidden, 2, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.depthwise(e1)
        d2 = self.dilated(d1)
        dec = self.dec1(d2 + e1)
        return self.out(dec)  # 返回 ab

# ============ 数据 ============
def collate_fn(batch):
    gray = torch.stack([torch.from_numpy(np.array(b["gray_image"]).astype(np.float32) / 255.0).unsqueeze(0) for b in batch])
    target = torch.stack([torch.from_numpy(np.array(b["target_image"]).astype(np.float32) / 255.0).permute(2, 0, 1) for b in batch])
    return {"gray": gray, "target": target}

print("📂 Loading dataset...")
ds = load_dataset("parquet", data_files={"train": str(DATA_DIR / "train.parquet")})
train_val = ds["train"].train_test_split(test_size=VAL_RATIO, seed=42)
train_loader = DataLoader(train_val["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(train_val["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
print(f"  Train: {len(train_val['train'])} | Val: {len(train_val['test'])}")

# ============ 模型 ============
model = ImprovedColorCNN_LAB(hidden=HIDDEN_CHANNELS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

n_params = sum(p.numel() for p in model.parameters())
print(f"📊 Model parameters: {n_params:,}")

# ============ Loss ============
class ColorBoostedLoss_LAB(nn.Module):
    """LAB 色彩空间的 Color-Boosted Loss"""
    def __init__(self, color_weight=5.0):
        super().__init__()
        self.color_weight = color_weight
    
    def forward(self, pred_ab, target_lab):
        # pred_ab: [B, 2, H, W], target_lab: [B, 3, H, W]
        target_ab = target_lab[:, 1:]  # 只取 ab
        # L1 loss on ab
        loss_ab = F.l1_loss(pred_ab, target_ab)
        # 可选：对 a 和 b 分别加权
        loss_a = F.l1_loss(pred_ab[:, 0:1], target_ab[:, 0:1])
        loss_b = F.l1_loss(pred_ab[:, 1:2], target_ab[:, 1:2])
        return loss_ab + self.color_weight * (loss_a + loss_b) / 2

criterion = ColorBoostedLoss_LAB(color_weight=5.0)

# ============ Metrics ============
def psnr(pred, target, eps=1e-8):
    mse = (pred - target).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))

def ssim_simple(pred, target, eps=1e-8):
    C1, C2 = 0.01**2, 0.03**2
    B, C, H, W = pred.shape
    pred_f, targ_f = pred.view(B, C, -1), target.view(B, C, -1)
    mu_x, mu_y = pred_f.mean(dim=-1), targ_f.mean(dim=-1)
    var_x, var_y = pred_f.var(dim=-1, unbiased=False), targ_f.var(dim=-1, unbiased=False)
    cov_xy = ((pred_f - mu_x.unsqueeze(-1)) * (targ_f - mu_y.unsqueeze(-1))).mean(dim=-1)
    ssim_c = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2) + eps)
    return ssim_c.mean(dim=1)

# ============ 训练循环 ============
print("\n" + "="*60)
print("🚀 开始训练 - LAB 色彩空间 + Color-Boosted Loss + LR Scheduling")
print("="*60)

best_psnr = 0
best_model_state = None

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss_sum = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Train"):
        gray, target = batch["gray"].to(DEVICE), batch["target"].to(DEVICE)
        # 转 LAB
        target_lab = rgb2lab(target)
        L_input = target_lab[:, 0:1]  # L 通道（与 gray 相同）
        target_ab = target_lab[:, 1:]  # ab 通道
        
        optimizer.zero_grad()
        pred_ab = model(gray)
        loss = criterion(pred_ab, target_lab)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
    
    train_loss = train_loss_sum / len(train_loader)
    
    # Eval
    model.eval()
    total_psnr_val, total_ssim = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Eval"):
            gray, target = batch["gray"].to(DEVICE), batch["target"].to(DEVICE)
            target_lab = rgb2lab(target)
            pred_ab = model(gray)
            # 重建 RGB: L 用原图 + 预测的 ab
            pred_lab = torch.cat([target_lab[:, 0:1], pred_ab], dim=1)
            pred_rgb = lab2rgb(pred_lab)
            
            total_psnr_val += psnr(pred_rgb, target).sum().item()
            total_ssim += ssim_simple(pred_rgb, target).sum().item()
    
    n = len(val_loader.dataset)
    val_psnr, val_ssim = total_psnr_val / n, total_ssim / n
    
    # LR scheduling
    scheduler.step(train_loss)
    
    print(f"\nEpoch {epoch}/{EPOCHS}: Train Loss={train_loss:.4f} | Val PSNR={val_psnr:.2f} dB | SSIM={val_ssim:.3f}")
    
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        best_model_state = model.state_dict().copy()
        print(f"  💾 Best model saved! (PSNR: {best_psnr:.2f} dB)")

# 保存最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), OUTPUT_DIR / "model_lab_balanced.pt")
    print(f"\n✅ Best model saved to {OUTPUT_DIR / 'model_lab_balanced.pt'}")

# 保存指标
metrics = {
    "model": "ImprovedColorCNN_LAB",
    "features": [
        "LAB Color Space (predict ab only)",
        "Depthwise Separable Conv",
        "Dilated Conv (dilation=2)",
        "Color-Boosted Loss (5x)",
        "LR Scheduling (ReduceLROnPlateau)"
    ],
    "val_psnr": val_psnr,
    "val_ssim": val_ssim,
    "best_psnr": best_psnr,
    "epochs": EPOCHS,
    "hidden": HIDDEN_CHANNELS,
    "params": n_params
}

with open(OUTPUT_DIR / "metrics_lab_balanced.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n📋 Metrics saved to {OUTPUT_DIR / 'metrics_lab_balanced.json'}")
print("="*60)
print(f"✅ 训练完成！Best PSNR: {best_psnr:.2f} dB")
print("="*60)
