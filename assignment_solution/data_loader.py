import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class NeuralNavigatorDataset(Dataset):
    def __init__(self, root_dir, split="train", max_text_len=5):
        self.root_dir = root_dir
        self.split = split
        self.max_text_len = max_text_len

        if split == "train":
            self.image_dir = os.path.join(root_dir, "data/images")
            self.ann_dir = os.path.join(root_dir, "data/annotations")
        else:
            self.image_dir = os.path.join(root_dir, "test_data/images")
            self.ann_dir = os.path.join(root_dir, "test_data/annotations")

        self.files = sorted(os.listdir(self.ann_dir))

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # vocabulary
        self.vocab = {
            "go": 0,
            "to": 1,
            "red": 2,
            "blue": 3,
            "green": 4,
            "circle": 5,
            "square": 6,
            "triangle": 7
        }

        self.pad_id = len(self.vocab)

    def encode_text(self, text):
        tokens = text.lower().replace(".", "").split()
        ids = [self.vocab[t] for t in tokens if t in self.vocab]

        # pad / truncate
        if len(ids) < self.max_text_len:
            ids += [self.pad_id] * (self.max_text_len - len(ids))
        else:
            ids = ids[:self.max_text_len]

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ann_path = os.path.join(self.ann_dir, self.files[idx])
        with open(ann_path) as f:
            ann = json.load(f)

        img_path = os.path.join(self.image_dir, ann["image_file"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        text = self.encode_text(ann["text"])

        if self.split == "train":
            # normalize path to [0,1]
            path = torch.tensor(ann["path"], dtype=torch.float32) / 128.0
            return image, text, path
        else:
            return image, text, ann["image_file"]
