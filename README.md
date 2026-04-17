# 🔍 Surface Crack Detection using CNN (PyTorch)

A deep learning project that classifies concrete surface images as **cracked** or **intact** using a custom Convolutional Neural Network built with PyTorch.

---

## 📌 Overview

Surface cracks in concrete structures can signal serious structural integrity issues. This project automates crack detection using a CNN trained on the [Surface Crack Detection dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) from Kaggle — enabling fast, accurate, and scalable visual inspection.

---

## 🧠 Model Architecture

A custom CNN (`CNN`) with three convolutional blocks followed by a fully connected classifier:

| Block | Layers |
|---|---|
| Conv Block 1 | Conv2d(3→64) × 2, BatchNorm, ReLU, MaxPool, Dropout(0.2) |
| Conv Block 2 | Conv2d(64→128) × 2, BatchNorm, ReLU, MaxPool, Dropout(0.3) |
| Conv Block 3 | Conv2d(128→256) × 2, BatchNorm, ReLU, MaxPool, Dropout(0.4) |
| Classifier | Flatten → Linear(512) → BatchNorm → ReLU → Dropout(0.5) → Linear(1) → Sigmoid |

Binary classification output: `1 = Positive (crack detected)`, `0 = Negative (no crack)`

---

## 🗂️ Dataset

- **Source:** [Kaggle – Surface Crack Detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) by Arun RK
- **Classes:** `Positive` (cracked), `Negative` (intact)
- **Split:** 80% training / 20% validation

---

## ⚙️ Data Preprocessing & Augmentation

**Training transforms:**
- Resize to 124px → RandomCrop(110)
- Random horizontal & vertical flip
- Color jitter (brightness & contrast ±0.2)
- Normalize: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)

**Validation transforms:**
- Resize to 124px → CenterCrop(110)
- Normalize (same as above)

---

## 🚀 Training

| Parameter | Value |
|---|---|
| Optimizer | Adam (default lr) |
| Loss Function | Binary Cross Entropy (BCELoss) |
| Epochs | 5 |
| Batch Size | 64 |
| Device | CUDA (GPU) / CPU fallback |

The best model weights (highest validation accuracy) are saved automatically to `best_paramiters.pt`.

---

## 🔎 Inference

Use the `check_for_crack()` function to run inference on any image from the dataset:

```python
check_for_crack("Positive/00660.jpg")   # → 'Positive'
check_for_crack("Negative/00427.jpg")   # → 'Negative'
```

The function loads the image, applies validation transforms, runs the model, and displays the image alongside the prediction.

---

## 📦 Requirements

```
torch
torchvision
kagglehub
Pillow
matplotlib
numpy
pandas
```

Install with:

```bash
pip install torch torchvision kagglehub Pillow matplotlib numpy pandas
```

---

## 🏃 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/surface-crack-detection.git
   cd surface-crack-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** (handled automatically in the notebook via `kagglehub`)

4. **Run the notebook**
   ```bash
   jupyter notebook Surface-crack-detection.ipynb
   ```

---

## 📁 Project Structure

```
surface-crack-detection/
├── Surface-crack-detection.ipynb   # Main notebook
├── best_paramiters.pt              # Saved best model weights (generated after training)
├── requirements.txt
└── README.md
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
