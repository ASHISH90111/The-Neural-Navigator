# The-Neural-Navigator
 Build a Neural Network that acts like a "Smart GPS". Your model must take two inputs: a 2D image (a map with colored shapes) and a text command (e.g., "Go to the Red Circle"), and output a sequence of (x, y) coordinates representing the path to the target.
📌 Problem Statement

Given:

A 128×128 RGB image containing simple geometric shapes

A text instruction describing the target

Predict:

A sequence of 10 (x, y) coordinates forming a path from the center of the image to the target.

📂 Project Structure
assignment_solution/
│
├── data_loader.py        # Dataset & preprocessing
├── model.py              # Vision + Text fusion model
├── train.py              # Training loop with scheduler
├── predict.py            # Inference & visualization
├── requirements.txt
├── outputs/
│   ├── training_loss.png
│   ├── pred_0.png ...
└── README.md

🧠 Model Architecture
1. Vision Encoder

A CNN extracts spatial features from the 128×128 RGB image:

3 convolution blocks

Adaptive average pooling

Output flattened feature vector

2. Text Encoder

Learnable embedding for instruction tokens

Padding-aware embedding

Mean pooling over tokens

3. Fusion

The image and text features are concatenated and passed through a fully connected network.

4. Decoder

A regression head predicts 10 (x, y) coordinates representing the navigation path.

🏗 Architecture Diagram (Conceptual)
Image ──▶ CNN ──┐
                ├── Concatenate ── FC Layers ── Path (10×2)
Text  ──▶ Embed ┘

🧪 Training Details
Loss Function
Total Loss = MSE(path, ground_truth)
           + 0.1 × Smoothness Loss


Smoothness loss penalizes abrupt direction changes:

(path[:, 1:] - path[:, :-1])²


This produces more realistic trajectories.

Optimizer & Scheduler

Optimizer: Adam

Learning Rate: 1e-3

Scheduler: StepLR (step=10, gamma=0.7)

📉 Training Behavior

Rapid initial convergence

Stable decreasing loss

Smooth convergence curve

No divergence or instability

A training-loss plot is saved automatically:

outputs/training_loss.png

🧠 Inference

Run:

python predict.py


This:

Loads the trained model

Runs inference on test images

Draws predicted paths

Saves output images to outputs/

Example output:

✔ Path points move toward the correct target
✔ Smooth trajectory
✔ Correct semantic grounding (color + shape)

⚠️ Challenges & Solutions
1. Model checkpoint incompatibility

Problem:
Changing network layers caused size mismatch errors when loading old checkpoints.

Solution:
Implemented a safe loading mechanism that loads only compatible weights and skips mismatched layers. This allowed training to resume without crashing.

2. Text padding caused index errors

Problem:
Text sequences had different lengths, causing embedding index errors.

Solution:
Added a padding token and updated embedding size accordingly. Padding index was ignored during learning.

3. Noisy / jagged predicted paths

Problem:
Initial predictions had sharp, unrealistic direction changes.

Solution:
Added a smoothness regularization term that penalizes large step-to-step changes.

4. Training instability during early epochs

Solution:
Used a learning-rate scheduler to stabilize convergence.

📊 Accuracy / Performance

Training loss decreases smoothly and converges

Model consistently predicts paths toward correct targets

Visual outputs are coherent and interpretable

Demonstrates correct multimodal reasoning

The goal was not perfect geometric precision but stable reasoning and correct directional intent.

✅ Key Features Implemented

✔ Custom PyTorch Dataset
✔ Vision encoder (CNN)
✔ Text embedding encoder
✔ Multi-modal fusion
✔ Regression-based path prediction
✔ Smoothness regularization
✔ Learning rate scheduler
✔ Resume-safe checkpoint loading
✔ Prediction visualization
✔ Clean modular code

🧾 Requirements
torch
torchvision
numpy
opencv-python
matplotlib
tqdm
Pillow

🚀 How to Run
Train
python train.py

Predict
python predict.py

🎯 Final Notes

This project demonstrates:

Multi-modal learning

Practical debugging skills

Model iteration & refinement

Stable training practices

Clean engineering structure

It aligns with real-world robotics ML workflows where perception, language, and control must be combined effectively.
