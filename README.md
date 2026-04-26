# Understanding the Latent Space of Generative Models

Project members: **Akshita Potdar** and **Christian Newman-Sanders**
Lead by: **Andres Ramirez-Jaime**

## Overview

This project explores the latent spaces learned by generative neural networks (VAE and GAN) on MNIST and FFHQ, then uses those latent spaces to solve **inverse problems** — recovering clean images from degraded measurements.

We solve the classic inverse problem:

```
y = A(x) + ε
```

by optimizing in the latent space of a pretrained generator:

```
z* = argmin_z  L1( A(G(z)) − y )
```

Once we find `z*`, the recovered image is `G(z*)`.

## Repo structure

```
.
├── models/                    # Generator and VAE definitions
│   ├── vae_mnist.py
│   ├── gan_mnist.py
│   ├── vae_ffhq.py
│   └── stylegan_ffhq.py       # Wrapper around pretrained StyleGAN2
├── training/                  # Training scripts
│   ├── train_vae_mnist.py
│   ├── train_gan_mnist.py
│   └── train_vae_ffhq.py
├── inverse_problems/          # The core contribution
│   ├── operators.py           # Forward operators A() — masks, downsample, noise
│   ├── solver.py              # Latent-space optimization
│   ├── demo_mnist.py          # End-to-end MNIST demo
│   └── demo_ffhq.py           # End-to-end FFHQ demo with StyleGAN2
├── utils/
│   ├── visualize.py           # Latent-manifold plots, iteration strips
│   └── metrics.py             # FID and MSE
├── notebooks/                 # Exploratory Jupyter notebooks
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

For the FFHQ StyleGAN2 demo, also install:

```bash
pip install stylegan2-pytorch pytorch-fid
```

Then download a pretrained FFHQ StyleGAN2 (or train your own) and place it in `./pretrained/stylegan2-ffhq/`.

## Reproducing the presentation results

### 1. Train MNIST models

```bash
python -m training.train_vae_mnist --epochs 30
python -m training.train_gan_mnist --epochs 50
```

### 2. Latent manifold plots (slides 9 and 10)

```python
import torch
from models import VAE_MNIST, GAN_Generator_MNIST
from utils import plot_latent_manifold

# VAE manifold
vae = VAE_MNIST(latent_dim=2)
vae.load_state_dict(torch.load("./checkpoints/vae_mnist.pt"))
plot_latent_manifold(vae.decode, title="2D Latent Space Manifold of MNIST Digits (VAE)",
                     save_path="vae_manifold.png")

# GAN manifold
gan = GAN_Generator_MNIST(latent_dim=2)
gan.load_state_dict(torch.load("./checkpoints/gan_mnist_G.pt"))
gan.eval()
plot_latent_manifold(lambda z: (gan(z) + 1) / 2,  # rescale tanh output to [0,1]
                     title="2D Latent Space Manifold of MNIST Digits (GAN)",
                     save_path="gan_manifold.png")
```

### 3. FFHQ inverse problems (slides 14 and 15)

```bash
python -m inverse_problems.demo_ffhq \
    --stylegan-dir ./pretrained/stylegan2-ffhq \
    --target ./assets/target_face.png \
    --out-dir ./results
```

This produces:
- `identity_iters.png` — pure inversion (no degradation)
- `box_inpaint_iters.png` — large box occlusion
- `random_inpaint_iters.png` — scattered missing pixels
- `super_res_iters.png` — 4× super resolution
- `denoise_iters.png` — Gaussian noise removal
- `qualitative_grid.png` — side-by-side comparisons
- `metrics.txt` — MSE for each problem

### 4. MNIST inverse problems

```bash
python -m inverse_problems.demo_mnist \
    --gan-checkpoint ./checkpoints/gan_mnist_G.pt \
    --target-idx 0
```

## Key results from the presentation

### Generative model quality (FID — lower is better)

| Model | FID  |
|-------|------|
| GAN   | 27.3 |
| VAE   | 145.16 |

The VAE produces blurry faces (visible in the qualitative results) because of the inherent reconstruction–KL trade-off. The GAN produces sharper, more realistic faces.

### Inverse problem quality (MSE — lower is better)

| Problem               | GAN MSE  |
|-----------------------|----------|
| Super-resolution      | 0.00253  |
| Inpainting (large)    | 0.00258  |
| Inpainting (small)    | 0.00213  |
| Denoising             | 0.00283  |

## Conclusions

- The latent space of a generative model is rich enough to act as a strong **prior** for solving inverse problems, even though the model was never trained on those tasks specifically.
- **Solution quality is bounded by the generator's expressiveness.** Our basic GAN gives reasonable results on FFHQ; using StyleGAN2 or a Diffusion model would improve sharpness, identity preservation, and out-of-distribution behavior considerably.

## References

- Bora et al., *Compressed Sensing using Generative Models* (2017) — the foundational paper for this approach.
- Karras et al., *A Style-Based Generator Architecture for Generative Adversarial Networks* (StyleGAN, 2019).
- Kingma & Welling, *Auto-Encoding Variational Bayes* (VAE, 2013).
- Goodfellow et al., *Generative Adversarial Nets* (2014).

## Contributions

- **Akshita** — Data collection, slides, FFHQ VAE training, inverse problem solution
- **Christian** — Data collection, slides, FFHQ GAN training, inverse problem solution
