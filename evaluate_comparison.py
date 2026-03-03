#!/usr/bin/env python3
"""评估改进模型 vs 基线模型"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path
import numpy as np
import json

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

print(f"🚀 Device: {DEVICE}")

# === Models ===
class TinyColorCNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

class ImprovedColorCNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)
        )
        self.dilated = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(hidden, 3, 1), nn.Sigmoid())
    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.depthwise(e1)
        d2 = self.dilated(d1)
        return self.out(self.dec1(d2 + e1))

# === Data ===
def collate_fn(batch):
    gray = torch.stack([torch.from_numpy(np.array(b["gray_image"]).astype(np.float32) / 255.0).unsqueeze(0) for b in batch])
    target = torch.stack([torch.from_numpy(np.array(b["target_image"]).astype(np.float32) / 255.0).permute(2, 0, 1) for b in batch])
    return {"gray": gray, "target": target}

ds = load_dataset("parquet", data_files={"train": str(DATA_DIR / "train.parquet")})
val = ds["train"].train_test_split(test_size=0.1, seed=42)["test"]
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# === Metrics ===
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

def evaluate(model, loader):
    model.eval()
    total_psnr, total_ssim = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            gray, target = batch["gray"].to(DEVICE), batch["target"].to(DEVICE)
            pred = model(gray)
            total_psnr += psnr(pred, target).sum().item()
            total_ssim += ssim_simple(pred, target).sum().item()
    n = len(loader.dataset)
    return total_psnr / n, total_ssim / n

# === Evaluate ===
print("\n=== 基线模型 (TinyColorCNN) ===")
model_base = TinyColorCNN(64).to(DEVICE)
model_base.load_state_dict(torch.load(OUTPUT_DIR / "model.pt", weights_only=True, map_location=DEVICE))
psnr_b, ssim_b = evaluate(model_base, val_loader)
print(f"PSNR: {psnr_b:.2f} dB | SSIM: {ssim_b:.3f}")

print("\n=== 改进模型 (ImprovedColorCNN) ===")
model_imp = ImprovedColorCNN(64).to(DEVICE)
model_imp.load_state_dict(torch.load(OUTPUT_DIR / "model_improved.pt", weights_only=True, map_location=DEVICE))
psnr_i, ssim_i = evaluate(model_imp, val_loader)
print(f"PSNR: {psnr_i:.2f} dB | SSIM: {ssim_i:.3f}")

print("\n" + "="*60)
print(f"📊 对比结果")
print(f"  基线 → PSNR: {psnr_b:.2f} dB, SSIM: {ssim_b:.3f}")
print(f"  改进 → PSNR: {psnr_i:.2f} dB, SSIM: {ssim_i:.3f}")
print(f"  提升 → PSNR: {psnr_i-psnr_b:+.2f} dB, SSIM: {ssim_i-ssim_b:+.3f}")
print("="*60)

# Save
with open(OUTPUT_DIR / "comparison.json", "w") as f:
    json.dump({"baseline": {"psnr": psnr_b, "ssim": ssim_b}, "improved": {"psnr": psnr_i, "ssim": ssim_i}, "delta": {"psnr": psnr_i-psnr_b, "ssim": ssim_i-ssim_b}}, f, indent=2)

print(f"\n✅ Saved to {OUTPUT_DIR / 'comparison.json'}")
