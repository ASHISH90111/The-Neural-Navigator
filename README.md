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

├── data_loader.py # Dataset & preprocessing

├── model.py # Vision + text fusion model

├── train.py # Training pipeline

├── predict.py # Inference & visualization

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

### What was the hardest part of this assignment?

The hardest part was designing a training and evaluation setup that looked realistic and meaningful, even though the dataset is small, synthetic, and highly deterministic.
Because the images, shapes, and target coordinates follow a very regular structure, the model quickly learns an almost perfect mapping. The challenge was not just achieving low loss, but understanding why the model converges so fast, how to measure performance properly, and how to avoid misleading metrics.

---

### Describe a specific bug you encountered and how you debugged it

One major issue I faced was state_dict loading errors when resuming training after modifying the model architecture.

Problem

After changing the network (adding/removing layers), loading a previously saved checkpoint caused errors like:
```scss
size mismatch for fc.weight
unexpected key(s) in state_dict

```
This happened because PyTorch requires exact tensor shape matches when loading model weights.

How I debugged it

I inspected the error messages to identify which layers had mismatched shapes.

Printed and compared keys from:

model.state_dict()

checkpoint .pth file

Identified that only some layers were still compatible.

Fix

I implemented safe checkpoint loading, where:

Only parameters with matching names and shapes are loaded.

Incompatible layers are skipped.

Training continues normally with new layers initialized.

This allowed me to:

Resume training without crashes

Experiment with architecture changes safely

Preserve already learned representations

---

##📊 Performance Summary

- Mean endpoint error converges to ~0.05–0.1 pixels
- Success rate ≈ 95% using a 2-pixel threshold
- Evaluation is performed on the same synthetic distribution as training
- High accuracy is expected due to the deterministic structure of the dataset
- Predictions are smooth and visually consistent with the target direction

This metric reflects endpoint proximity rather than classification accuracy.


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



