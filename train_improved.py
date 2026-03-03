#!/usr/bin/env python3
"""
Lab 4 CNN Colorization - Improved Version
基于 Discussion 经验改进：
1. Depthwise Separable Convs (更高效)
2. Color-Boosted Loss (5x color penalty)
3. Dilated Convs (增加 receptive field)
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

# ==================== Configuration ====================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
HIDDEN_CHANNELS = 64
EPOCHS = 5
LEARNING_RATE = 2e-3
VAL_RATIO = 0.1

print(f"🚀 Using device: {DEVICE}")
print(f"📊 Training: {EPOCHS} epochs, hidden={HIDDEN_CHANNELS}")

# ==================== Model: Improved ColorCNN ====================
class ImprovedColorCNN(nn.Module):
    """
    改进版 CNN，基于 Discussion 经验：
    1. Depthwise Separable Convs - 更高效
    2. Dilated Convs - 增加 receptive field
    3. 保留 skip connection 思想
    """
    def __init__(self, hidden=64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        
        # Depthwise Separable Conv
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),  # Depthwise
            nn.Conv2d(hidden, hidden, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        
        # Dilated Conv (增加 receptive field)
        self.dilated = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        
        # Output
        self.out = nn.Sequential(
            nn.Conv2d(hidden, 3, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        
        # Depthwise Separable
        d1 = self.depthwise(e1)
        
        # Dilated
        d2 = self.dilated(d1)
        
        # Decoder (with skip connection from e1)
        dec = self.dec1(d2 + e1)  # Simple skip connection
        
        return self.out(dec)

# ==================== Data Loading ====================
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

# ==================== Model & Training ====================
model = ImprovedColorCNN(hidden=HIDDEN_CHANNELS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

n_params = sum(p.numel() for p in model.parameters())
print(f"📊 Model parameters: {n_params:,}")

# ==================== Color-Boosted Loss ====================
class ColorBoostedLoss(nn.Module):
    """
    基于 Discussion 经验：
    L1 Loss + 5x color penalty
    分离 luminance 和 chrominance，对 color 部分施加更大 penalty
    """
    def __init__(self, color_weight=5.0):
        super().__init__()
        self.color_weight = color_weight
    
    def forward(self, pred, target):
        # 转换到 YCbCr 色彩空间（近似）
        # Y = 0.299*R + 0.587*G + 0.114*B
        # Cb, Cr = color channels
        
        # Luminance loss (权重 1x)
        pred_luma = pred.mean(dim=1, keepdim=True)
        target_luma = target.mean(dim=1, keepdim=True)
        loss_luma = F.l1_loss(pred_luma, target_luma)
        
        # Chrominance loss (权重 5x)
        loss_chroma = F.l1_loss(pred, target)
        
        # 总 loss = luma + 5 * chroma
        return loss_luma + self.color_weight * loss_chroma

criterion = ColorBoostedLoss(color_weight=5.0)

# ==================== Metrics ====================
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

# ==================== Training Loop ====================
best_psnr = 0
for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Train"):
        gray, target = batch["gray"].to(DEVICE), batch["target"].to(DEVICE)
        optimizer.zero_grad()
        pred = model(gray)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    
    # Eval
    model.eval()
    total_loss, total_psnr_val, total_ssim = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Eval"):
            gray, target = batch["gray"].to(DEVICE), batch["target"].to(DEVICE)
            pred = model(gray)
            total_loss += criterion(pred, target).item() * gray.size(0)
            total_psnr_val += psnr(pred, target).sum().item()
            total_ssim += ssim_simple(pred, target).sum().item()
    
    n = len(val_loader.dataset)
    val_loss, val_psnr, val_ssim = total_loss/n, total_psnr_val/n, total_ssim/n
    
    print(f"\nEpoch {epoch}/{EPOCHS}: Train L1={loss.item():.4f} | Val L1={val_loss:.4f} | PSNR={val_psnr:.2f} dB | SSIM={val_ssim:.3f}")
    
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(model.state_dict(), OUTPUT_DIR / "model_improved.pt")
        print(f"  💾 Best model saved!")

# Save metrics
metrics = {
    "model": "ImprovedColorCNN",
    "features": ["Depthwise Separable Conv", "Dilated Conv", "Color-Boosted Loss (5x)"],
    "val_l1": val_loss,
    "val_psnr": val_psnr,
    "val_ssim": val_ssim,
    "best_psnr": best_psnr,
    "epochs": EPOCHS,
    "hidden": HIDDEN_CHANNELS
}

with open(OUTPUT_DIR / "metrics_improved.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Done! Best PSNR: {best_psnr:.2f} dB")
print(f"📋 Metrics saved to {OUTPUT_DIR / 'metrics_improved.json'}")
