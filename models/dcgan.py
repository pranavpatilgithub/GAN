import torch
import torch.nn as nn

LATENT_DIM = 100
IMG_CHANNELS = 1


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


def load_generator(checkpoint_path, device):
    gen = Generator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["generator_state_dict"])
    gen.eval()
    return gen


def generate(generator, num_images, device, **kwargs):
    z = torch.randn(num_images, LATENT_DIM, 1, 1, device=device)
    with torch.no_grad():
        imgs = generator(z)
    # Denormalize from [-1, 1] to [0, 1]
    imgs = (imgs + 1) / 2
    return imgs
