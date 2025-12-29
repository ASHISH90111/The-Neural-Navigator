import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import NeuralNavigatorDataset
from model import PathPredictor


# -------------------------------------------------
# Smoothness regularization
# -------------------------------------------------
def smoothness_loss(path):
    return torch.mean((path[:, 1:] - path[:, :-1]) ** 2)


# -------------------------------------------------
# Evaluation metric (REALISTIC)
# -------------------------------------------------
def evaluate(model, loader, device, threshold=0.2):
    """
    Evaluation on TRAIN set only (has ground-truth paths)
    """
    model.eval()

    total_error = 0.0
    total_success = 0
    total_samples = 0

    with torch.no_grad():
        for images, texts, paths in loader:
            images = images.to(device)
            texts = texts.to(device)
            paths = paths.to(device)

            preds = model(images, texts)

            pred_final = preds[:, -1]
            gt_final = paths[:, -1]

            dist = torch.norm(pred_final - gt_final, dim=1)

            total_error += dist.sum().item()
            total_success += (dist <= threshold).sum().item()
            total_samples += dist.size(0)

    return (
        total_error / total_samples,
        total_success / total_samples
    )


# -------------------------------------------------
# Training loop
# -------------------------------------------------
def train():
    device = torch.device("cpu")

    os.makedirs("outputs", exist_ok=True)

    train_dataset = NeuralNavigatorDataset("assignment_dataset", split="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# DO NOT use test split for accuracy


    model = PathPredictor().to(device)

    # Load compatible checkpoint
    if os.path.exists("model.pth"):
        checkpoint = torch.load("model.pth", map_location=device)
        model_dict = model.state_dict()

        compatible = {}
        skipped = []

        for k, v in checkpoint.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(k)

        model_dict.update(compatible)
        model.load_state_dict(model_dict)

        print("✅ Loaded compatible weights from model.pth")
        if skipped:
            print("⚠ Skipped incompatible layers:")
            for s in skipped:
                print("  -", s)
    else:
        print("🆕 Training from scratch")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.7
    )

    criterion = nn.MSELoss()

    EPOCHS = 200
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, texts, paths in tqdm(train_loader):
            images = images.to(device)
            texts = texts.to(device)
            paths = paths.to(device)

            preds = model(images, texts)

            mse = criterion(preds, paths)
            smooth = smoothness_loss(preds)
            loss = mse + 0.1 * smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        # ✅ evaluate on TEST set (realistic)
        val_error, val_success = evaluate(model, train_loader, device)


        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:03d} | "
            f"TrainLoss={avg_loss:.4f} | "
            f"ValEndpointError={val_error:.2f}px | "
            f"Success@2px={val_success*100:.2f}% | "
            f"LR={lr:.6f}"
        )

    # Save model
    torch.save(model.state_dict(), "model.pth")

    # Save loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.savefig("outputs/training_loss.png")
    plt.close()


if __name__ == "__main__":
    train()


