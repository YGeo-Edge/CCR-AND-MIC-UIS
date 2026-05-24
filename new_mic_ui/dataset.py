"""
Dataset that reads images from a folder-per-class structure.
Works for both local paths and S3-downloaded data.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = 448) -> transforms.Compose:
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        classes: List[str],
        image_size: int = 448,
        transform: Optional[transforms.Compose] = None,
        max_per_class: Optional[int] = None,
    ):
        self.root = Path(root)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform or build_transform(image_size)
        self.samples: List[Tuple[Path, int]] = []

        for cls in classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                print(f"  Warning: {cls_dir} missing — skipping")
                continue
            files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if max_per_class:
                files = files[:max_per_class]
            for f in files:
                self.samples.append((f, self.class_to_idx[cls]))

        counts = {cls: sum(1 for _, idx in self.samples if idx == self.class_to_idx[cls]) for cls in classes}
        print(f"Dataset loaded: {len(self.samples)} images across {len(classes)} classes")
        for cls, n in counts.items():
            print(f"  {cls}: {n}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        try:
            img = Image.open(path)
            pixel_values = self.transform(img)
        except Exception as e:
            print(f"  Warning: failed to load {path}: {e} — using blank image")
            pixel_values = torch.zeros(3, 448, 448)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }
