import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dim = 100
num_classes = 10
img_shape = (1,28,28)
batch_size = 64
epochs = 500
lr = 0.0002

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

dataset = MNIST(
    root = "/kaggle/working",
    train = False,
    transform = transform,
    download=True
)   

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        img = self.model(x)
        return img.view(img.size(0), *img_shape)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(28 * 28 + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img = img.view(img.size(0), -1)
        c = self.label_emb(labels)
        x = torch.cat([img, c], dim=1)
        return self.model(x)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


import os
fixed_z = torch.randn(10, latent_dim, device=device)
fixed_labels = torch.arange(0, 10, device=device)


os.makedirs("cgan_outputs", exist_ok=True)

def save_generated_images(epoch):
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(fixed_z, fixed_labels).detach().cpu()

    grid = torchvision.utils.make_grid(
        gen_imgs,
        nrow=10,
        normalize=True
    )

    plt.figure(figsize=(12, 2))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(f"Epoch {epoch}")
    plt.show()

    save_image(
        gen_imgs,
        f"cgan_outputs/epoch_{epoch}.png",
        nrow=10,
        normalize=True
    )

    generator.train()


for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        real_imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size_curr = real_imgs.size(0)

        valid = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        # -----------------
        # Train Discriminator
        # -----------------
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(
            discriminator(real_imgs, labels), valid
        )

        z = torch.randn(batch_size_curr, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (batch_size_curr,), device=device)
        gen_imgs = generator(z, gen_labels)

        fake_loss = adversarial_loss(
            discriminator(gen_imgs.detach(), gen_labels), fake
        )

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        g_loss = adversarial_loss(
            discriminator(gen_imgs, gen_labels), valid
        )
        g_loss.backward()
        optimizer_G.step()

        if i % 400 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{i}/{len(dataloader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )

    if (epoch + 1) in [1, 50, 100, 500, 1000]:
        save_generated_images(epoch + 1)
