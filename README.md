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

yaml
Copy code

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

yaml
Copy code

---

## 🧪 Training

### Loss Function

The total loss is:

Total Loss = MSE Loss + 0.1 × Smoothness Loss

lua
Copy code

Where smoothness loss penalizes sharp direction changes:

```python
(path[:, 1:] - path[:, :-1])²
This helps generate smoother and more realistic trajectories.

Optimization
Optimizer: Adam

Learning rate: 1e-3

Scheduler: StepLR(step_size=10, gamma=0.7)

📉 Training Behavior
Rapid initial loss decrease

Stable convergence

Smooth training curve

No exploding gradients

Stable long-term training

The training loss curve is saved automatically:

bash
Copy code
outputs/training_loss.png
🖼 Inference
Run:

bash
Copy code
python predict.py
This will:

Load the trained model

Run inference on test images

Draw predicted paths

Save visual outputs inside outputs/

⚠️ Challenges & Solutions
1. Model checkpoint incompatibility
Problem:
Changing the architecture caused state_dict size mismatch errors when loading old checkpoints.

Solution:
Implemented safe loading by only loading compatible weights and skipping mismatched layers. This allowed continued training without crashes.

2. Text padding caused embedding index errors
Problem:
Text instructions have variable length, which caused out-of-range errors in embedding layers.

Solution:
Added a padding token and updated the embedding layer to support padding safely.

3. Jagged / noisy predicted paths
Problem:
Initial predictions had abrupt direction changes.

Solution:
Added a smoothness regularization term to penalize sharp path changes.

4. Training instability
Problem:
Loss oscillated after several epochs.

Solution:
Added a learning-rate scheduler to gradually reduce the learning rate and stabilize training.

📊 Performance Summary
Training loss decreases smoothly

Stable convergence behavior

Correct directional movement toward target

Generalizes to unseen samples

Produces visually meaningful trajectories

The objective is not pixel-perfect accuracy, but correct reasoning and stable learning behavior.

📦 Requirements
nginx
Copy code
torch
torchvision
numpy
opencv-python
matplotlib
tqdm
Pillow
Install dependencies:

bash
Copy code
pip install -r requirements.txt
🚀 How to Run
Train the model
bash
Copy code
python train.py
Run inference
bash
Copy code
python predict.py
✅ Summary
This project demonstrates:

Multi-modal learning (vision + language)

CNN-based visual perception

Text embedding and fusion

Regression-based trajectory prediction

Debugging and model iteration

Stable training with scheduling

Clean modular PyTorch code

The implementation reflects practical challenges faced in robotics and embodied AI systems.

✅ Ready for submission

yaml
Copy code

---

If you want, I can also help you with:

✅ Email reply to the recruiter  
✅ Short interview explanation (spoken version)  
✅ Improve prediction quality further  
✅ Add validation accuracy metric  
✅ Polish GitHub formatting  
✅ Add diagrams  

Just tell me what you want next.
