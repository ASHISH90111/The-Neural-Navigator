import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from model import PathPredictor
from data_loader import NeuralNavigatorDataset


def draw_path(image, path):
    img = np.array(image)

    for i in range(len(path) - 1):
        p1 = tuple(path[i].astype(int))
        p2 = tuple(path[i + 1].astype(int))
        cv2.line(img, p1, p2, (255, 0, 0), 2)

    return img


def main():
    device = torch.device("cpu")

    os.makedirs("outputs", exist_ok=True)

    dataset = NeuralNavigatorDataset("assignment_dataset", split="test")

    model = PathPredictor().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    for i in range(min(5, len(dataset))):
        image, text, filename = dataset[i]

        with torch.no_grad():
            pred = model(
                image.unsqueeze(0).to(device),
                text.unsqueeze(0).to(device)
            )

            # 🔥 denormalize from [0,1] → [0,128]
            path = pred.squeeze(0).cpu().numpy() * 128.0

        raw_img = Image.open(
            f"assignment_dataset/test_data/images/{filename}"
        ).convert("RGB")

        vis = draw_path(raw_img, path)

        cv2.imwrite(
            f"outputs/pred_{i}.png",
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        )

        print(f"Saved: outputs/pred_{i}.png")


if __name__ == "__main__":
    main()
