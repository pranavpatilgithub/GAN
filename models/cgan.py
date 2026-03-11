import torch
import torch.nn as nn

LATENT_DIM = 100
NUM_CLASSES = 10
IMG_SHAPE = (1, 28, 28)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        img = self.model(x)
        return img.view(img.size(0), *IMG_SHAPE)


def load_generator(checkpoint_path, device):
    gen = Generator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["generator_state_dict"])
    gen.eval()
    return gen


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.model = nn.Sequential(
            nn.Linear(28 * 28 + NUM_CLASSES, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        img = img.view(img.size(0), -1)
        c = self.label_emb(labels)
        x = torch.cat([img, c], dim=1)
        return self.model(x)


def load_discriminator(checkpoint_path, device):
    disc = Discriminator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    disc.load_state_dict(ckpt["discriminator_state_dict"])
    disc.eval()
    return disc


def predict(discriminator, img_tensor, device, label=0, **kwargs):
    img_tensor = img_tensor.to(device)
    labels = torch.full((img_tensor.size(0),), label, dtype=torch.long, device=device)
    with torch.no_grad():
        score = discriminator(img_tensor, labels)
    return score.item()


def generate(generator, num_images, device, label=0, **kwargs):
    z = torch.randn(num_images, LATENT_DIM, device=device)
    labels = torch.full((num_images,), label, dtype=torch.long, device=device)
    with torch.no_grad():
        imgs = generator(z, labels)
    # Denormalize from [-1, 1] to [0, 1]
    imgs = (imgs + 1) / 2
    return imgs
