"""
Train the MNIST GAN.

Usage:
    python -m training.train_gan_mnist --epochs 50 --batch-size 128
"""
import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import GAN_Generator_MNIST, GAN_Discriminator_MNIST


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize to [-1, 1] to match generator's tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    G = GAN_Generator_MNIST(latent_dim=args.latent_dim).to(device)
    D = GAN_Discriminator_MNIST().to(device)

    opt_g = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    for epoch in range(1, args.epochs + 1):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            bs = real.size(0)
            real_lbl = torch.ones(bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)

            # ---- Train D ----
            opt_d.zero_grad()
            d_real = D(real)
            loss_d_real = bce(d_real, real_lbl)

            z = torch.randn(bs, args.latent_dim, device=device)
            fake = G(z)
            d_fake = D(fake.detach())
            loss_d_fake = bce(d_fake, fake_lbl)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # ---- Train G ----
            opt_g.zero_grad()
            d_fake_for_g = D(fake)
            loss_g = bce(d_fake_for_g, real_lbl)
            loss_g.backward()
            opt_g.step()

        print(f"Epoch {epoch:3d} | D loss {loss_d.item():.4f} | G loss {loss_g.item():.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(args.save_dir, "gan_mnist_G.pt"))
    torch.save(D.state_dict(), os.path.join(args.save_dir, "gan_mnist_D.pt"))
    print(f"Saved G and D to {args.save_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
