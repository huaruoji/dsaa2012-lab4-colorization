# DSAA2012 Lab 4: Image Colorization with Balanced LAB Model 🎨

[![GitHub](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/pytorch-2.7+-red.svg)](https://pytorch.org/)

A balanced approach to image colorization combining insights from course discussions and literature research.

---

## 🎯 Motivation

Based on the Lab 4 Open Discussion, several classmates (Hengrui WU, Yehua HUANG) shared valuable insights:

- **L1 Loss Problem**: Leads to conservative, desaturated predictions ("gray/brown bias")
- **Color-Boosted Loss**: Applying 5x penalty on color channels helps
- **Architecture**: Depthwise + Dilated convolutions improve efficiency and receptive field
- **Open Question**: Has anyone tried LAB color space?

This project implements a **balanced solution** combining these ideas.

---

## 🚀 Improvements

| Component | Source | Implementation |
|-----------|--------|----------------|
| **LAB Color Space** | Literature | Predict ab channels only, keep L from input |
| **Color-Boosted Loss (5x)** | Hengrui WU | 5x penalty on chrominance errors |
| **Depthwise Separable Conv** | Hengrui WU | Efficient feature extraction |
| **Dilated Conv (d=2)** | Hengrui WU | Increased receptive field |
| **LR Scheduling** | Literature | ReduceLROnPlateau for stability |

---

## 📊 Results

### Validation Set Performance

| Model | PSNR (dB) | SSIM | Parameters | Notes |
|-------|-----------|------|------------|-------|
| **Baseline (TinyColorCNN)** | 16.72 | 0.769 | 38K | 2-layer Conv, L1 Loss |
| **Improved (RGB)** | 16.79 | 0.778 | 80K | Depthwise+Dilated+Color Loss |
| **LAB Balanced Model** | **21.69** | **0.917** | 80K | LAB + 5x + LR Scheduling |

### Einstein Test Image

| Metric | Value | Assessment |
|--------|-------|------------|
| **PSNR** | 24.44 dB | Good pixel-level accuracy |
| **Visual Quality** | Moderate | Colors are more natural than baseline, but skin tones could be more accurate |
| **Comparison** | Better than baseline | Noticeable improvement over initial TinyColorCNN |

**Honest Assessment:**
- ✅ **Significant metric improvement** (+4.97 dB PSNR vs baseline)
- ✅ **More vibrant colors** than L1-only baseline
- ⚠️ **Not perfect** - some color artifacts remain, especially in skin tones
- ⚠️ **Generalization** - works better on some images than others

---

## 🏗️ Architecture

```
Input (1ch Gray)
    ↓
Encoder: Conv2d(1→64) + BN + ReLU
    ↓
Depthwise Separable: DW-Conv + PW-Conv
    ↓
Dilated Conv (dilation=2)
    ↓
Decoder: Conv2d(64→64) + BN + ReLU (+ skip connection)
    ↓
Output: Conv2d(64→2)  # ab channels
    ↓
Combine: L (from input) + ab (predicted) → RGB
```

**Key Design:**
- Only predict ab channels (2 outputs instead of 3)
- Luminance (L) comes directly from input grayscale
- Model focuses purely on color prediction

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/huaruoji/dsaa2012-lab4-colorization.git
cd dsaa2012-lab4-colorization

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Training

```bash
python train_lab_balanced.py
```

**Configuration:**
- Epochs: 5
- Batch Size: 64
- Hidden Channels: 64
- Learning Rate: 2e-3 (with ReduceLROnPlateau)
- Color Weight: 5.0

### Evaluation

```bash
python evaluate_comparison.py
```

---

## 📁 Project Structure

```
lab4/
├── train_lab_balanced.py      # LAB balanced model training
├── train_improved.py          # RGB improved model (baseline for comparison)
├── train.py                   # Original TinyColorCNN baseline
├── evaluate_comparison.py     # Model comparison script
├── README.md                  # This file
├── pyproject.toml             # Project configuration
├── requirements.txt           # Dependencies
├── data/                      # Dataset (gitignored)
│   ├── train.parquet
│   ├── test.parquet
│   └── solution.parquet
└── outputs/                   # Models and results (gitignored)
    ├── model_lab_balanced.pt
    ├── metrics_lab_balanced.json
    └── *.png (visualizations)
```

---

## 💡 Key Findings

1. **LAB Color Space is Effective**
   - PSNR improved from 16.72 → 21.69 dB (+4.97 dB)
   - Separating luminance from chrominance simplifies the task

2. **Preserving L Channel Matters**
   - Avoids relearning brightness information
   - Model can focus on color prediction

3. **Color-Boosted Loss Works**
   - Addresses the "safe average" problem of L1 loss
   - Produces more vibrant colors

4. **LR Scheduling Stabilizes Training**
   - Prevents overfitting in later epochs
   - Helps converge to better minima

---

## 🔬 Comparison with Discussion Insights

### Hengrui WU's Approach
- Used Color-Boosted Loss (5x) → **Adopted**
- Depthwise + Dilated Convs → **Adopted**
- RGB space → **Improved to LAB**
- Reported: PSNR 16.98 dB, Aesthetic 3.15

### Yehua HUANG's Approach
- TinyColorCNN baseline → **Used as starting point**
- Suggested: Perceptual Loss, U-Net → **Future work**
- Suggested: Multimodal (text prompts) → **Out of scope**

### This Work
- **Combined** insights from multiple posts
- **Extended** with LAB color space
- **Achieved** higher PSNR (21.69 dB) but trade-offs in visual quality

---

## 📚 References

1. Hengrui WU. "Lab 4 Open Discussion: Black-and-White Image Colorization." DSAA2012, 2026.
2. Yehua HUANG. "Lab 4 Open Discussion: Black-and-White Image Colorization." DSAA2012, 2026.
3. Saeed Anwar et al. "Image colorization: A survey and dataset." *Information Fusion*, Vol. 114, Oct 2024. DOI: [10.1016/j.inffus.2024.102720](https://doi.org/10.1016/j.inffus.2024.102720)
4. Xiangcheng Du et al. "MultiColor: Image Colorization by Learning from Multiple Color Spaces." *ACM MM '24*, Oct 28, 2024. DOI: [10.1145/3664647.3680726](https://doi.org/10.1145/3664647.3680726)
5. Richard Zhang, Phillip Isola, Alexei A. Efros. "Colorful Image Colorization." *ECCV 2016*. arXiv: [1603.08511](https://arxiv.org/abs/1603.08511)

**Verification:** All references verified via Brave Search on Mar 4, 2026. ✅

---

## 🤝 Contributing

This is a course project for DSAA2012 (Deep Learning) at HKUST-GZ.

Feel free to:
- 🐛 Open issues for bugs or questions
- 💡 Suggest improvements
- 🍴 Fork and experiment

---

## 📝 Future Work

- [ ] **Perceptual Loss (VGG)** - Add high-level semantic loss
- [ ] **Aesthetic Score Evaluation** - Use SigLIP for perceptual quality
- [ ] **U-Net Architecture** - Test skip connections
- [ ] **More Color Spaces** - Compare LAB, LUV, YUV
- [ ] **Data Augmentation** - Test impact on generalization
- [ ] **User Study** - Evaluate visual quality beyond PSNR/SSIM

---

## 📄 License

MIT License - This is a course project for educational purposes.

---

## 👨‍💻 Author

DSAA2012 Lab 4 Project  
HKUST-GZ, Spring 2026

**Acknowledgments:** Thanks to classmates for sharing insights in the course discussion!
