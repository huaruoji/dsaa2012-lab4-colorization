# Lab 4: Image Colorization - Balanced LAB Model 🎨

## 实验总结 (2026-03-03)

基于 Canvas 课程 Discussion 中其他同学的经验，实现了平衡的图像彩色化模型。

---

## 🎯 改进方案

综合了 Discussion 中 Hengrui WU、Yehua HUANG 等同学的经验：

| 改进点 | 来源 | 实现 |
|--------|------|------|
| **LAB 色彩空间** | 文献调研 | 只预测 ab 通道，L 用原图 |
| **Color-Boosted Loss (5x)** | Hengrui WU | 对 color 施加更大 penalty |
| **Depthwise Separable Conv** | Hengrui WU | 高效特征提取 |
| **Dilated Conv** | Hengrui WU | 增加 receptive field |
| **LR Scheduling** | 文献调研 | ReduceLROnPlateau |

---

## 📊 实验结果

### 模型对比

| 模型 | PSNR (dB) | SSIM | 参数量 | 特点 |
|------|-----------|------|--------|------|
| **基线 (TinyColorCNN)** | 16.72 | 0.769 | 38K | 2 层 Conv, L1 Loss |
| **改进 (RGB)** | 16.79 | 0.778 | 80K | Depthwise+Dilated+Color Loss |
| **LAB 平衡模型** | **21.69** | **0.917** | 80K | LAB + 5x + LR |

### 爱因斯坦测试

- **PSNR: 24.44 dB** (vs 原图)
- 颜色自然，饱和度高
- 皮肤色调还原准确

---

## 🚀 使用方法

### 训练
```bash
cd lab4
source .venv/bin/activate
python train_lab_balanced.py
```

### 配置
- **Epochs:** 5
- **Batch Size:** 64
- **Hidden Channels:** 64
- **Learning Rate:** 2e-3 (with scheduling)
- **Color Weight:** 5.0

---

## 📁 文件结构

```
lab4/
├── train_lab_balanced.py    # LAB 平衡模型训练脚本
├── train_improved.py        # RGB 改进版训练脚本
├── train.py                 # 基线模型训练脚本
├── evaluate_comparison.py   # 模型对比评估脚本
├── data/                    # 数据集 (已 gitignore)
├── outputs/                 # 模型和结果 (已 gitignore)
│   ├── model_lab_balanced.pt
│   ├── metrics_lab_balanced.json
│   └── einstein_comparison_lab.png
└── README_LAB.md            # 本文件
```

---

## 💡 关键发现

1. **LAB 色彩空间效果显著** - PSNR 从 16.7 提升到 21.7 dB
2. **L 通道保留很重要** - 避免模型重新学习亮度信息
3. **Color-Boosted Loss 有效** - 强制模型输出鲜艳颜色
4. **LR Scheduling 稳定训练** - 避免后期过拟合

---

## 📚 参考文献

1. Hengrui WU. Lab 4 Open Discussion. DSAA2012, 2026.
2. "Image colorization: A survey and dataset", 2024.
3. "MultiColor: Image Colorization by Learning from Multiple Color Spaces", ACM MM 2024.

---

## 👥 贡献

- 基于 DSAA2012 Lab 4 课程项目
- Discussion 经验总结 + 自主改进
- 2026-03-03 完成

---

## 📝 待改进

- [ ] 添加 Aesthetic Score 评估
- [ ] 尝试 Perceptual Loss (VGG)
- [ ] 测试更多样本
- [ ] 对比 U-Net 架构
