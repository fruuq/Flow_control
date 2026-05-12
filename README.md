# 🏟️ Flow Control — AI-Based Crowd Management System

> Mustafa Firas Mustafa Mathhar 

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![University of Jordan](https://img.shields.io/badge/University-of%20Jordan-blue)](https://www.ju.edu.jo/)

---

## 📋 Overview

**Flow Control** is an AI-based crowd management system designed for large-scale sports events such as the **2026 FIFA World Cup**. The system monitors crowd density in real time, predicts congestion before it peaks, and recommends concrete interventions to stadium operators.

The system is built around **three integrated layers**:

| Layer | Description |
|---|---|
| 🎥 **Perception** | Estimates crowd density from CCTV and drone feeds |
| 🔮 **Prediction** | Forecasts zone-level congestion 5 minutes ahead |
| 🚨 **Decision Support** | Recommends gate balancing, rerouting, and staff redeployment |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        INPUT                            │
│  CCTV Ground  │  Drone Cameras  │  Gate Cameras  │ BLE │
└───────┬───────┴────────┬────────┴───────┬─────────┴──┬──┘
        ▼                ▼                ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                   PERCEPTION LAYER                      │
│  CSRNet-UNet  │    CSRNet     │  YOLO-CROWD  │ Aggreg. │
│  (Stands)     │   (Aerial)    │  (Entrances) │  Zone   │
└───────────────────────────────────┬─────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────┐
│                   PREDICTION LAYER                      │
│         LSTM Forecaster  │  Uncertainty Estimator       │
└───────────────────────────────────┬─────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────┐
│                 DECISION SUPPORT LAYER                  │
│  Gate Balancing │ Crowd Rerouting │ Staff Redeployment  │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 Models

### 1. CSRNet-UNet — Football Supporters Crowd (FSC)
- **Backbone**: VGG-16 (ImageNet pretrained)
- **Decoder**: U-Net with skip connections
- **Input**: 512×512 | **Output**: 256×256 density map
- **Dataset**: FSC-Set (~3,600 images across 3 sub-datasets)
- **Best MAE**: 13.23 | **Best RMSE**: 42.35

### 2. CSRNet — Drone Aerial Surveillance
- **Backbone**: VGG-16 + Dilated CNN backend (dilation=2)
- **Input**: 384×384 | **Output**: 48×48 density map
- **Dataset**: 112 aerial sequences (~3,360 frames)
- **Test MAE**: 26.31 | **Test RMSE**: 33.82

### 3. YOLO-CROWD — Access Control & Entrances
- **Backbone**: YOLOv5 (COCO pretrained)
- **Input**: 640×640 | **Task**: Head detection
- **Dataset**: Roboflow Crowd Counting Dataset (2,898 images)
- **Precision**: 0.771 | **Recall**: 0.442 | **mAP@0.5**: 0.461

---

## 📊 Results

### Backbone Comparison (FSC Validation Set)

| Method | Best MAE | Best RMSE | Δ MAE |
|---|---|---|---|
| EfficientNet-B0 + U-Net | 20.84 | 50.31 | — |
| **CSRNet-UNet (Ours)** | **13.23** | **42.35** | **−36.5%** |

### All Models

| Model | Dataset | MAE | RMSE |
|---|---|---|---|
| CSRNet-UNet | FSC (Val) | **13.23** | **42.35** |
| CSRNet | Drone (Test) | 26.31 | 33.82 |
| YOLO-CROWD | Roboflow (Test) | 78.60 | 182.0 |

### Weather Augmentation Ablation

| Configuration | Best MAE | Best RMSE |
|---|---|---|
| Without weather aug ✅ | **13.23** | **42.35** |
| With weather aug | 17.17 | 49.23 |

> Aggressive weather augmentation increased MAE by **+29.7%**, indicating domain mismatch with the FSC dataset's predominantly daytime footage.

---

## 📁 Project Structure

```
flow_control/
│
├── 📓 crowd-counter-fcsnet.ipynb     # CSRNet-UNet training on FSC dataset
├── 📓 drone-work.ipynb               # CSRNet training on drone dataset
│
├── 🏋️ models/
│   ├── best_model.pth                # Best CSRNet-UNet checkpoint (no weather aug)
│   └── models_weather/
│       └── best_model.pth            # Best CSRNet-UNet checkpoint (with weather aug)
│
├── 📊 data/
│   ├── FSC/
│   │   ├── Supporters/               # ~1,800 images + pre-computed .npy maps
│   │   ├── Supporters-Team/          # ~1,600 images + .mat annotations (19 teams)
│   │   └── Empty_Scenes/             # 202 empty stadium images
│   └── Drone/                        # 112 sequences, ~3,360 frames
│
└── 📄 README.md
```

---

## 🚀 Quick Start

### Requirements

```bash
pip install torch==2.6.0 torchvision opencv-python scipy tqdm matplotlib
```

### Run FSC Crowd Counter

```python
import torch
from model import CSRNetUNet

# Load best checkpoint
model = CSRNetUNet().cuda()
checkpoint = torch.load("models/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    density_map = model(image.cuda())
    count = density_map.sum().item()
    print(f"Estimated count: {count:.1f} persons")
```

### Decision Support Layer

```python
# Alert thresholds
THRESHOLD_CRITICAL = 500   # > 500 persons → Open all gates
THRESHOLD_HIGH     = 300   # 300–500      → Redirect crowd flow
THRESHOLD_MODERATE = 150   # 150–300      → Increase monitoring

def get_alert(count):
    if count > THRESHOLD_CRITICAL:  return "🚨 CRITICAL"
    elif count > THRESHOLD_HIGH:    return "⚠️  HIGH"
    elif count > THRESHOLD_MODERATE:return "📢 MODERATE"
    else:                           return "✅ NORMAL"
```

---

## 🌦️ Weather Augmentation

The system includes a `WeatherAugmentation` pipeline with three stochastic transforms:

| Effect | Method | Probability |
|---|---|---|
| **Fog** | Linear blend with white overlay (α ∈ [0.3, 0.6]) | 0.3 |
| **Night** | Darkening + blue-channel shift | 0.3 |
| **Rain** | Random line segments at 20–30° angle | 0.3 |

---

## 📦 Datasets

| Dataset | Images | Split | Source |
|---|---|---|---|
| FSC – Supporters | ~1,800 | 80/10/10 | [FSC-Set](https://doi.org/10.1109/ACCESS.2022.3144607) |
| FSC – Supporters-Team | ~1,600 | 80/10/10 | FSC-Set |
| FSC – Empty Scenes | 202 | 80/10/10 | FSC-Set |
| Drone Dataset | ~3,360 | 80/20 | VisDrone-CC2021 |
| YOLO-CROWD | 2,898 | 79/13/8 | [Roboflow](https://universe.roboflow.com/crowd-dataset/crowd-counting-dataset-w3o7w) |

---

## 🖥️ Hardware

| Environment | CPU | GPU | Used For |
|---|---|---|---|
| Local (Windows) | AMD Ryzen 7 | RTX 2060 6GB | YOLO training, prototyping |
| Kaggle Cloud | — | T4 16GB | CSRNet-UNet, CSRNet training |

---

## 📄 Paper

This project is accompanied by a full IEEE-format conference paper:

> **"An AI-Based Crowd Management System for Large-Scale Sports Events: A 2026 FIFA World Cup Case Study"**  
> Mustafa Firas Mustafa Mathhar, Zaid Abdullah Yousef Awad, Yazan Naesr Mohammad Algazi  
> Department of Artificial Intelligence, University of Jordan, 2026

---

## 👥 Authors

| Name | GitHub |
|---|---|
| Mustafa Firas Mustafa Mathhar | [@fruuq](https://github.com/fruuq) |
| Zaid Abdullah Yousef Awad | — |
| Yazan Naesr Mohammad Algazi | — |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
