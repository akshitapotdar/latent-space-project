"""
Train the FFHQ VAE on 64x64 face crops.

Expects FFHQ images in args.data_dir organized as either:
  - data_dir/<class>/*.png  (ImageFolder-style, any single class folder works)
  - data_dir/*.png          (use --flat)

You can grab a small subset for testing from:
  https://github.com/NVlabs/ffhq-dataset

Usage:
    python -m training.train_vae_ffhq --data-dir ./ffhq64 --epochs 50
"""
import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from models import VAE_FFHQ, vae_ffhq_loss


class FlatImageFolder(Dataset):
    """For when all images sit directly in one folder, no class subdirs."""

    def __init__(self, root, transform=None):
        self.root = root
        self.files = sorted(
            f for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),  # [0, 1] — matches sigmoid output
    ])

    if args.flat:
        dataset = FlatImageFolder(args.data_dir, transform=transform)
    else:
        dataset = ImageFolder(args.data_dir, transform=transform)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    model = VAE_FFHQ(latent_dim=args.latent_dim, image_size=args.image_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_ffhq_loss(recon, x, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg = running / len(dataset)
        print(f"Epoch {epoch:3d} | avg loss/img {avg:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "vae_ffhq.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--flat", action="store_true",
                   help="data_dir contains images directly, not class subfolders")
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
