<div align="center">

# 🌍 DeepGlobe Land Cover Classification

**Semantic Segmentation of Satellite Imagery using Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-DeepGlobe-20BEFF.svg)](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)

_Automated land cover classification from high-resolution satellite imagery for environmental monitoring and urban planning._

[Features](#-features) • [Installation](#-installation) • [Dataset](#-dataset) • [Usage](#-usage) • [Results](#-results) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Acknowledgments](#-acknowledgments)

## 🎯 Overview

This project implements a **semantic segmentation pipeline** for land cover classification using the DeepGlobe dataset. The model classifies each pixel in satellite images into one of **7 land cover classes**:

| Class          | Color                                                                   | Description           |
| -------------- | ----------------------------------------------------------------------- | --------------------- |
| 🏙️ Urban       | ![#00FFFF](https://via.placeholder.com/15/00FFFF/000000?text=+) Cyan    | Built-up areas, roads |
| 🌾 Agriculture | ![#FFFF00](https://via.placeholder.com/15/FFFF00/000000?text=+) Yellow  | Farmland, crops       |
| 🌿 Rangeland   | ![#FF00FF](https://via.placeholder.com/15/FF00FF/000000?text=+) Magenta | Grassland, shrubs     |
| 🌲 Forest      | ![#00FF00](https://via.placeholder.com/15/00FF00/000000?text=+) Green   | Trees, woodland       |
| 💧 Water       | ![#0000FF](https://via.placeholder.com/15/0000FF/000000?text=+) Blue    | Lakes, rivers, ocean  |
| 🏜️ Barren      | ![#FFFFFF](https://via.placeholder.com/15/FFFFFF/000000?text=+) White   | Desert, bare soil     |
| ❓ Unknown     | ![#000000](https://via.placeholder.com/15/000000/000000?text=+) Black   | Unlabeled regions     |

### Key Statistics

> **Dataset Size:** 803 training images • **Image Resolution:** 2448×2448 pixels
> **Total Pixels:** ~4.8 billion labeled pixels • **Class Imbalance:** 17.4:1 ratio

## ✨ Features

- ✅ **Exploratory Data Analysis** with comprehensive visualizations
- ✅ **Class Imbalance Handling** with computed weights (Water: 2.46, Agriculture: 0.14)
- ✅ **Efficient Data Pipeline** with RGB→ID conversion (3× memory reduction)
- 🚧 **Custom PyTorch Dataset** with augmentations
- 🚧 **U-Net/DeepLabv3+** architecture with weighted loss
- 🚧 **Training & Evaluation** with mIoU metrics
- 🚧 **Inference Pipeline** for new satellite images

## 📁 Project Structure

```
ml-deepglobe/
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb    # ✅ EDA & Analysis
│   ├── 02_test_dataloader.ipynb     # 🚧 Pipeline Testing
│   └── 03_evaluate_results.ipynb    # 🚧 Model Evaluation
├── 🐍 src/
│   ├── data/
│   │   ├── dataset.py               # PyTorch Dataset
│   │   ├── transforms.py            # Augmentations
│   │   └── mask_conversion.py       # RGB↔ID conversion
│   ├── models/
│   │   ├── unet.py                  # U-Net architecture
│   │   └── deeplabv3.py             # DeepLabv3+
│   └── utils/
│       ├── metrics.py               # mIoU, per-class IoU
│       └── visualization.py         # Plotting utilities
├── 📊 outputs/
│   ├── figures/                     # Visualizations
│   └── models/                      # Saved checkpoints
├── 📄 requirements.txt              # Python dependencies
└── 📄 README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended
- Kaggle API credentials (for dataset download)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-deepglobe.git
cd ml-deepglobe

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Jupyter kernel
python -m ipykernel install --user --name=ml-deepglobe
```

<details>
<summary><b>🔧 Troubleshooting Installation Issues</b></summary>

**CUDA Issues:**

```bash
# Verify CUDA version
nvidia-smi

# Install PyTorch with specific CUDA version
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Kaggle API Setup:**

```bash
# 1. Get your API token from https://www.kaggle.com/settings
# 2. Place kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

</details>

## 📊 Dataset

### DeepGlobe Land Cover Classification

- **Source:** [Kaggle - DeepGlobe Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
- **Original Challenge:** [DeepGlobe CVPR 2018](http://deepglobe.org/)
- **License:** Competition use only (check Kaggle for details)

### Download & Prepare

```bash
# Download dataset (requires Kaggle API)
kaggle datasets download -d balraj98/deepglobe-land-cover-classification-dataset

# Extract to data directory
unzip deepglobe-land-cover-classification-dataset.zip -d data/raw/

# Verify structure
ls data/raw/train/  # Should show *_sat.jpg and *_mask.png files
```

### Class Distribution

| Class       | Pixels | Percentage | Weight         |
| ----------- | ------ | ---------- | -------------- |
| Agriculture | 2.78B  | **57.74%** | 0.14           |
| Forest      | 537M   | 11.16%     | 0.73           |
| Urban       | 520M   | 10.80%     | 0.75           |
| Rangeland   | 408M   | 8.48%      | 0.96           |
| Barren      | 407M   | 8.45%      | 0.96           |
| Water       | 159M   | **3.31%**  | **2.46**       |
| Unknown     | 2.5M   | 0.05%      | 0.00 (ignored) |

> ⚠️ **Note:** Agriculture is **17.4× more common** than Water, requiring weighted loss functions.

## 💻 Usage

### 1️⃣ Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**What you'll find:**

- 📈 Class distribution analysis
- ⚖️ Computed class weights for balanced training
- 🔄 RGB↔ID conversion validation
- 📊 Visualization of sample predictions

### 2️⃣ Train Model (Coming Soon)

```bash
python train.py --config configs/unet_baseline.yaml
```

### 3️⃣ Inference (Coming Soon)

```bash
python inference.py --image path/to/satellite.jpg --checkpoint outputs/models/best.pth
```

## 🏗️ Model Architecture

<details>
<summary><b>U-Net (Baseline)</b></summary>

- **Encoder:** ResNet-50 pretrained on ImageNet
- **Decoder:** Standard U-Net decoder with skip connections
- **Output:** 7-channel logits (one per class)
- **Loss:** Weighted CrossEntropyLoss with class weights
- **Optimizer:** AdamW with learning rate 1e-4

</details>

<details>
<summary><b>DeepLabv3+ (Advanced)</b></summary>

- **Backbone:** ResNet-101 with atrous convolutions
- **ASPP:** Atrous Spatial Pyramid Pooling
- **Decoder:** Lightweight decoder with skip connections
- **Output:** 7-channel logits
- **Loss:** Weighted CrossEntropyLoss + Dice Loss

</details>

## 📈 Results

> 🚧 **Coming Soon** - Model training in progress

### Expected Performance

| Metric          | Target |
| --------------- | ------ |
| Mean IoU        | > 60%  |
| Water IoU       | > 50%  |
| Agriculture IoU | > 70%  |

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepGlobe Challenge** organizers for the dataset
- **Segmentation Models PyTorch** library by Pavel Yakubovskiy
- **PyTorch** and **Albumentations** teams for excellent tools

---
