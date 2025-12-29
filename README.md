# Neural Navigator – Multi-Modal Path Prediction

This project implements a **multi-modal neural network ("Smart GPS")** that predicts a navigation path from a 2D map image and a natural-language instruction such as *"Go to the Red Circle"*.

The model fuses **visual features from an image** and **semantic features from text**, then outputs a sequence of `(x, y)` coordinates representing a path to the target.

---

## 📌 Problem Statement

Given:
- A 128×128 RGB image containing colored shapes
- A text instruction describing the target

Predict:
- A sequence of 10 `(x, y)` coordinates forming a path from the image center to the target.

---

## 📂 Project Structure

---

## 🧠 Model Architecture

### Vision Encoder
- CNN with 3 convolution layers
- Extracts spatial visual features
- Output flattened feature vector

### Text Encoder
- Learnable word embeddings
- Padding-safe embedding
- Mean pooling over tokens

### Fusion
Image and text embeddings are concatenated before prediction.

### Decoder
Fully connected network predicting **10 (x, y)** coordinates. 

---

## 🧪 Training

### Loss Function

```text
Total Loss = MSE(path, ground_truth)
           + 0.1 × Smoothness Loss
(path[:, 1:] - path[:, :-1])²


---

## 🧩 Architecture Overview


