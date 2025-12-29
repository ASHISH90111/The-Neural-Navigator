import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=8, emb_dim=32):
        super().__init__()

        # +1 for PAD token
        self.embedding = nn.Embedding(
            vocab_size + 1,
            emb_dim,
            padding_idx=vocab_size
        )

    def forward(self, x):
        emb = self.embedding(x)
        return emb.mean(dim=1)


class PathPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision = VisionEncoder()
        self.text = TextEncoder()

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4 + 32, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 20)
        )

    def forward(self, image, text):
        v = self.vision(image)
        t = self.text(text)
        fused = torch.cat([v, t], dim=1)
        out = self.fc(fused)
        return out.view(-1, 10, 2)
