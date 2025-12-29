# Neural Navigator – Multi-Modal Path Prediction

This project implements a **multi-modal neural network ("Smart GPS")** that predicts a navigation path from a 2D image and a natural-language instruction such as:

> "Go to the Red Circle"

The model takes:
- a **128×128 RGB image**
- a **text command**

and outputs a sequence of **(x, y) coordinates** representing a path to the target.

---

## 📌 Problem Statement

Given:
- A 2D map image with colored shapes  
- A text instruction describing the target  

Predict:
- A sequence of 10 `(x, y)` points forming a navigation path from the image center to the target object.

---

## 📂 Project Structure

assignment_solution/
│
├── data_loader.py        # Dataset & preprocessing  
├── model.py              # Vision + Text fusion model  
├── train.py              # Training pipeline  
├── predict.py            # Inference + visualization  
├── requirements.txt  
├── outputs/
│   ├── training_loss.png
│   ├── pred_0.png
│   ├── pred_1.png
│   └── ...
└── README.md

---

## 🧠 Model Architecture

### Vision Encoder
- Convolutional Neural Network (CNN)
- Extracts spatial features from the image
- Output flattened feature vector

### Text Encoder
- Learnable word embeddings  
- Padding-safe embedding  
- Mean pooling over token embeddings  

### Fusion
Image and text embeddings are concatenated before prediction.

### Decoder
A fully connected network predicts **10 (x, y)** coordinate pairs.

---

## 🧩 Architecture Overview

Image ──▶ CNN ──┐  
                ├── Concatenate ── FC Layers ── Path (10 × 2)  
Text  ──▶ Embed ┘  

---

## 🧪 Training

### Loss Function

Total loss is defined as:

MSE Loss + 0.1 × Smoothness Loss

Smoothness loss penalizes sharp direction changes:

```python
(path[:, 1:] - path[:, :-1]) ** 2
```

This encourages smoother and more realistic trajectories.

---

## ⚙️ Optimization

- Optimizer: **Adam**
- Learning rate: **1e-3**
- Scheduler: **StepLR(step_size=10, gamma=0.7)**

---

## 📉 Training Behavior

- Rapid initial loss decrease  
- Stable convergence  
- Smooth training curve  
- No exploding gradients  
- Stable long-term optimization  

Training loss is automatically saved to:

outputs/training_loss.png

---

## 🖼 Inference

Run:

```bash
python predict.py
```

This will:

- Load the trained model  
- Run inference on test images  
- Draw predicted paths  
- Save outputs inside `outputs/`

---

## ⚠️ Challenges & Solutions

### 1. Model checkpoint incompatibility

**Problem:**  
Changing architecture caused `state_dict` size mismatch errors.

**Solution:**  
Implemented safe checkpoint loading that only loads compatible weights and skips mismatched layers.

---

### 2. Text padding caused embedding index errors

**Problem:**  
Variable-length instructions caused out-of-range indices in embedding layers.

**Solution:**  
Added padding-safe embedding logic and consistent vocabulary handling.

---

### 3. Jagged / noisy predicted paths

**Problem:**  
Early predictions produced sharp or unstable trajectories.

**Solution:**  
Added a smoothness regularization term to penalize abrupt direction changes.

---

### 4. Training instability

**Problem:**  
Loss oscillated after several epochs.

**Solution:**  
Introduced a learning-rate scheduler to gradually reduce the learning rate and stabilize training.

---

## 📊 Performance Summary

- Training loss decreases smoothly  
- Stable convergence behavior  
- Correct directional movement toward target  
- Generalizes to unseen samples  
- Produces visually meaningful trajectories  

The goal is **not pixel-perfect accuracy**, but correct reasoning and stable learning behavior.

---

## 📦 Requirements

```
torch
torchvision
numpy
opencv-python
matplotlib
tqdm
Pillow
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Train the model
```bash
python train.py
```

### Run inference
```bash
python predict.py
```

---

## ✅ Summary

This project demonstrates:

- Multi-modal learning (vision + language)
- CNN-based visual perception
- Text embedding and fusion
- Regression-based trajectory prediction
