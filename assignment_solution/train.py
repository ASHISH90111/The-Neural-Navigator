import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import NeuralNavigatorDataset
from model import PathPredictor


def smoothness_loss(path):
    """
    Encourages smooth trajectories by penalizing sharp direction changes.
    path shape: (B, 10, 2)
    """
    return torch.mean((path[:, 1:] - path[:, :-1]) ** 2)


def train():
    device = torch.device("cpu")

    os.makedirs("outputs", exist_ok=True)

    dataset = NeuralNavigatorDataset("assignment_dataset", split="train")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PathPredictor().to(device)

    # Load previous checkpoint safely (ignore mismatched layers)
    if os.path.exists("model.pth"):
        checkpoint = torch.load("model.pth", map_location=device)
        model_dict = model.state_dict()

        compatible_state = {}
        skipped = []

        for k, v in checkpoint.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible_state[k] = v
            else:
                skipped.append(k)

        model_dict.update(compatible_state)
        model.load_state_dict(model_dict)

        print("✅ Loaded compatible weights from model.pth")
        if skipped:
            print("⚠ Skipped incompatible layers:")
            for s in skipped:
                print("   -", s)
    else:
        print("🆕 No existing model found — training from scratch")


    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ✅ Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.7
    )

    criterion = nn.MSELoss()

    losses = []
    EPOCHS = 200

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, texts, paths in tqdm(loader):
            images = images.to(device)
            texts = texts.to(device)
            paths = paths.to(device)

            preds = model(images, texts)

            mse_loss = criterion(preds, paths)
            smooth_loss = smoothness_loss(preds)

            loss = mse_loss + 0.1 * smooth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()  # ✅ decay learning rate

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} | LR = {current_lr:.6f}")

    # Save trained model
    torch.save(model.state_dict(), "model.pth")

    # Save loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE + Smoothness)")
    plt.title("Training Loss Curve")
    plt.savefig("outputs/training_loss.png")
    plt.close()


if __name__ == "__main__":
    train()

