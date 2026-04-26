"""
Train the MNIST VAE.

Usage:
    python -m training.train_vae_mnist --epochs 30 --batch-size 128
"""
import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import VAE_MNIST, vae_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    model = VAE_MNIST(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:3d} | avg loss {avg:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "vae_mnist.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
