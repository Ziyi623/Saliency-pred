import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch


class SaliencyDataset(Dataset):
    """
    Dataset for saliency prediction:
    - img_dir: path to Stimuli (RGB images)
    - map_dir: path to FIXATIONMAPS (grayscale saliency maps)
    - directory structure must match (e.g., Action/*.jpg)
    """
    def __init__(self, img_dir, map_dir, transform_img=None, transform_map=None):
        self.img_paths = []
        self.map_paths = []

        self.img_dir = img_dir
        self.map_dir = map_dir

        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(".jpg") or f.lower().endswith(".png"):
                    img_path = os.path.join(root, f)

                    # relative path under Stimuli
                    rel_path = os.path.relpath(img_path, img_dir)

                    # corresponding saliency-map path
                    map_path = os.path.join(map_dir, rel_path)

                    if os.path.exists(map_path):
                        self.img_paths.append(img_path)
                        self.map_paths.append(map_path)
                    else:
                        print(f"[Warning] GT not found for: {rel_path}")

        print(f"Loaded {len(self.img_paths)} samples from {img_dir}")

        self.transform_img = transform_img
        self.transform_map = transform_map

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        m = Image.open(self.map_paths[idx]).convert("L")

        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_map:
            m = self.transform_map(m)

        m = torch.clamp(m, 0., 1.)
        return img, m


def get_loaders(img_dir, map_dir, batch_size=4, img_size=(320, 180)):
    transform_img = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    transform_map = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
    ])

    train_ds = SaliencyDataset(img_dir, map_dir,
                               transform_img, transform_map)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,     # Windows 必须设为 0（否则多进程容易出错）
        pin_memory=True
    )

    return loader
