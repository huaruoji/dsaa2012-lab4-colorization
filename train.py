#!/usr/bin/env python3
"""
Lab 4 CNN Colorization - Minimal Implementation
DSAA2012 Deep Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# ==================== Configuration ====================
DATA_DIR = Path("/home/yuno/.openclaw/workspace-canvas/lab4/data")
OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
HIDDEN_CHANNELS = 32
EPOCHS = 1
LEARNING_RATE = 2e-3
VAL_RATIO = 0.1

print(f"🚀 Using device: {DEVICE}")

# ==================== Model ====================
class TinyColorCNN(nn.Module):
    """Minimal CNN for grayscale to RGB colorization"""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)

# ==================== Data Loading ====================
def load_data():
    """Load dataset from parquet files"""
    print("📂 Loading dataset...")
    
    ds = load_dataset(
        "parquet",
        data_files={
            "train": str(DATA_DIR / "train.parquet"),
            "test": str(DATA_DIR / "test.parquet"),
        },
    )
    
    # Create validation split
    train_val = ds["train"].train_test_split(test_size=VAL_RATIO, seed=42)
    ds_dict = {
        "train": train_val["train"],
        "val": train_val["test"],
        "test": ds["test"]
    }
    
    print(f"  Train: {len(ds_dict['train'])} samples")
    print(f"  Val: {len(ds_dict['val'])} samples")
    print(f"  Test: {len(ds_dict['test'])} samples")
    
    return ds_dict

def collate_fn(batch):
    """Custom collate function for batching"""
    ids = [b["id"] for b in batch]
    gray = torch.stack([torch.from_numpy(np.array(b["gray_image"]).astype(np.float32) / 255.0).unsqueeze(0) for b in batch])
    
    if "target_image" in batch[0] and batch[0]["target_image"] is not None:
        target = torch.stack([torch.from_numpy(np.array(b["target_image"]).astype(np.float32) / 255.0).permute(2, 0, 1) for b in batch])
        return {"id": ids, "gray": gray, "target": target}
    
    return {"id": ids, "gray": gray}

# ==================== Metrics ====================
def psnr(pred, target, eps=1e-8):
    """Calculate PSNR"""
    mse = (pred - target).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))

def ssim_simple(pred, target, eps=1e-8):
    """Simple SSIM implementation"""
    C1 = 0.01**2
    C2 = 0.03**2
    
    B, C, H, W = pred.shape
    pred_f = pred.view(B, C, -1)
    targ_f = target.view(B, C, -1)
    
    mu_x = pred_f.mean(dim=-1)
    mu_y = targ_f.mean(dim=-1)
    var_x = pred_f.var(dim=-1, unbiased=False)
    var_y = targ_f.var(dim=-1, unbiased=False)
    cov_xy = ((pred_f - mu_x.unsqueeze(-1)) * (targ_f - mu_y.unsqueeze(-1))).mean(dim=-1)
    
    ssim_c = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2) + eps
    )
    return ssim_c.mean(dim=1)

# ==================== Training ====================
def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Training"):
        gray = batch["gray"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(gray)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * gray.size(0)
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            gray = batch["gray"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            
            pred = model(gray)
            loss = F.l1_loss(pred, target)
            
            total_loss += loss.item() * gray.size(0)
            total_psnr += psnr(pred, target).sum().item()
            total_ssim += ssim_simple(pred, target).sum().item()
    
    n = len(loader.dataset)
    return total_loss / n, total_psnr / n, total_ssim / n

# ==================== Main ====================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load data
    ds = load_data()
    
    # Create dataloaders
    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(ds["val"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Initialize model
    model = TinyColorCNN(hidden=HIDDEN_CHANNELS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model parameters: {n_params:,}")
    
    # Training loop
    print(f"\n🏋️ Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_psnr, val_ssim = evaluate(model, val_loader)
        
        print(f"\nEpoch {epoch}/{EPOCHS}:")
        print(f"  Train L1: {train_loss:.4f}")
        print(f"  Val L1: {val_loss:.4f}")
        print(f"  Val PSNR: {val_psnr:.2f} dB")
        print(f"  Val SSIM: {val_ssim:.3f}")
    
    # Save metrics
    metrics = {
        "train_l1": train_loss,
        "val_l1": val_loss,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_channels": HIDDEN_CHANNELS,
        "learning_rate": LEARNING_RATE,
    }
    
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to {OUTPUT_DIR / 'metrics.json'}")
    
    # Generate predictions for visualization
    print("\n🎨 Generating predictions...")
    model.eval()
    
    # Get a few samples from validation set
    sample_batch = next(iter(val_loader))
    gray = sample_batch["gray"].to(DEVICE)
    target = sample_batch["target"]
    
    with torch.no_grad():
        pred = model(gray).cpu()
    
    # Save visualizations
    for i in range(min(5, gray.size(0))):
        gray_img = Image.fromarray((gray[i, 0].cpu().numpy() * 255).astype(np.uint8), mode='L')
        target_img = Image.fromarray((target[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode='RGB')
        pred_img = Image.fromarray((pred[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode='RGB')
        
        gray_img.save(OUTPUT_DIR / f"sample_{i}_gray.png")
        target_img.save(OUTPUT_DIR / f"sample_{i}_target.png")
        pred_img.save(OUTPUT_DIR / f"sample_{i}_pred.png")
    
    print(f"✅ Visualizations saved to {OUTPUT_DIR}/")
    
    # Print summary for Discussion post
    print("\n" + "="*50)
    print("📋 Discussion Post Summary")
    print("="*50)
    print(f"Model: TinyColorCNN ({n_params:,} parameters)")
    print(f"Hidden channels: {HIDDEN_CHANNELS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"\nResults:")
    print(f"  Val L1: {val_loss:.4f}")
    print(f"  Val PSNR: {val_psnr:.2f} dB")
    print(f"  Val SSIM: {val_ssim:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()
