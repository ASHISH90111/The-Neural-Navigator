# Neural Navigator – Multi-Modal Path Prediction

This project implements a **multi-modal neural network ("Smart GPS")** that predicts a navigation path from a 2D image and a natural-language instruction such as:

> “Go to the Red Circle”

The model takes:
- a **128×128 RGB image**, and  
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
├── data_loader.py # Dataset & preprocessing
├── model.py # Vision + Text fusion model
├── train.py # Training pipeline
├── predict.py # Inference + visualization
├── requirements.txt
├── outputs/
│ ├── training_loss.png
│ ├── pred_0.png
│ ├── pred_1.png
│ └── ...
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
The image and text embeddings are concatenated before prediction.

### Decoder
A fully connected network predicts **10 (x, y)** coordinate pairs.

---

## 🧩 Architecture Overview

Image ──▶ CNN ──┐
├── Concatenate ── FC Layers ── Path (10 × 2)
Text ──▶ Embed ┘


---

## 🧪 Training

### Loss Function

Total Loss = MSE Loss + 0.1 × Smoothness Loss


Smoothness loss penalizes sharp direction changes:

Smoothness loss penalizes sharp direction changes:

```python
(path[:, 1:] - path[:, :-1]) ** 2


✅ That’s it.

### Important rule (why this works)
- Opening fence: ```python  
- Closing fence: ```  
- Nothing else inside  
- Next text must start **after** the closing ``` on a new line

---

### Example in context (safe version)

```md
Smoothness loss penalizes sharp direction changes:

```python
(path[:, 1:] - path[:, :-1]) ** 2

