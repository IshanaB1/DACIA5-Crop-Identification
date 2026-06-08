# DACIA5 Crop Identification using Multi-Scale CNN and SE Attention

## Overview

This project focuses on crop classification using Sentinel-2 multispectral satellite imagery from the DACIA5 dataset. The objective is to identify agricultural crop types from 32×32 image patches containing 12 spectral bands.

The model combines Multi-Scale Convolutional Neural Networks (CNNs), Squeeze-and-Excitation (SE) Attention, Mixup Augmentation, and Label Smoothing to improve classification performance and generalization across different years of satellite observations.

---

## Problem Statement

Given a Sentinel-2 image patch, predict the crop type among the following classes:

| Class ID | Crop Type |
| -------- | --------- |
| 0        | Wheat     |
| 1        | Corn      |
| 2        | Peas      |
| 3        | Rapeseed  |
| 4        | Potato    |
| 5        | Sugarbeet |
| 6        | Alfalfa   |

The challenge evaluates model performance using the Q1 metric:

```text
Q1 = 0.5 × OA + 0.5 × AA
```

Where:

* OA = Overall Accuracy
* AA = Average Accuracy across all classes

---

## Dataset

### Source

DACIA5 Crop Identification Challenge Dataset

### Input Format

* Sentinel-2 Multispectral Imagery
* Patch Size: 32 × 32
* Spectral Bands: 12
* Training Years: 2020–2023
* Testing Year: 2024

### Data Structure

```text
dacia5_data/
└── patches/
    └── optical/
        ├── 2020/
        ├── 2021/
        ├── 2022/
        ├── 2023/
        └── 2024/
```

---

## Data Preprocessing

The following preprocessing steps are applied:

* Reflectance clipping to valid Sentinel-2 ranges
* Scaling pixel values to [0,1]
* Channel-wise standardization using StandardScaler
* Data augmentation:

  * Horizontal Flip
  * Vertical Flip
  * Random Rotation
  * Gaussian Noise Injection

---

## Model Architecture

The proposed architecture consists of three parallel feature extraction branches.

### Branch 1 – Local Spatial Features

* 3×3 Convolutions
* Residual Blocks
* SE Attention
* Max Pooling

Captures fine-grained spatial patterns.

### Branch 2 – Medium Scale Features

* 5×5 Convolutions
* Residual Blocks
* Adaptive Average Pooling

Captures larger crop structures and field patterns.

### Branch 3 – Spectral Feature Learning

* 1×1 Convolutions

Learns relationships between Sentinel-2 spectral bands.

### Feature Fusion

Features from all branches are concatenated and passed through a fully connected classification head.

```text
Input (12×32×32)
      │
 ┌────┼────┐
 │    │    │
3×3  5×5  1×1
 │    │    │
 └────┼────┘
      │
 Feature Fusion
      │
 SE Attention
      │
 Fully Connected Layers
      │
 7 Crop Classes
```

---

## Key Techniques

### Squeeze-and-Excitation (SE) Attention

SE blocks dynamically reweight feature channels, allowing the network to focus on the most informative spectral features.

### Residual Connections

Residual blocks improve gradient flow and enable deeper feature extraction.

### Mixup Augmentation

Training samples are combined using convex interpolation to improve generalization and reduce overfitting.

### Label Smoothing

Softens target labels to prevent overconfident predictions and improve calibration.

### OneCycle Learning Rate Scheduler

Accelerates convergence and improves final model performance.

---

## Training Configuration

| Parameter       | Value      |
| --------------- | ---------- |
| Optimizer       | AdamW      |
| Learning Rate   | 3e-4       |
| Batch Size      | 128        |
| Epochs          | 80         |
| Weight Decay    | 1e-4       |
| Label Smoothing | 0.05       |
| Mixup Alpha     | 0.3        |
| Scheduler       | OneCycleLR |

---

## Evaluation Metrics

### Overall Accuracy (OA)

```text
OA = Correct Predictions / Total Predictions
```

### Average Accuracy (AA)

Mean accuracy calculated independently for each crop class.

### Q1 Score

```text
Q1 = 0.5 × OA + 0.5 × AA
```

---

## Results

### Best Validation Performance

| Metric                | Score  |
| --------------------- | ------ |
| Overall Accuracy (OA) | XX.XX% |
| Average Accuracy (AA) | XX.XX% |
| Q1 Score              | XX.XX  |

Replace the values above with your final experimental results.

---

## Generated Outputs

### Model Checkpoint

```text
checkpoints/best_model.pth
```

### Confusion Matrix

```text
plots/confusion_matrix.png
```

### Per-Class Accuracy

```text
plots/per_class_accuracy.png
```

### Test Predictions

```text
checkpoints/test_predictions.npy
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/DACIA5-Crop-Identification.git
cd DACIA5-Crop-Identification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python train.py
```

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Scikit-Learn
* Rasterio
* SciPy
* Matplotlib
* Seaborn
* Sentinel-2 Remote Sensing Data

---

## Future Improvements

* Vision Transformer (ViT) based models
* Spectral Attention Networks
* Temporal Crop Monitoring
* Multi-Year Domain Adaptation
* Explainable AI for crop classification

---

## Author

Ishana Bharathi

Machine Learning | Deep Learning | Remote Sensing | Computer Vision
