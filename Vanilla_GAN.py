import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
import matplotlib.pyplot as plt
import os

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
latent_dim = 128
img_shape = (1, 28, 28)
batch_size = 16
lr = 0.0002
epochs = 500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

TARGET_DIGIT = 8   # change to 0-9

full_dataset = MNIST(root="./data", train=True, transform=transform, download=True)

indices = [i for i, (_, label) in enumerate(full_dataset) if label == TARGET_DIGIT]

from torch.utils.data import Subset
dataset = Subset(full_dataset, indices)

print(f"Training on digit '{TARGET_DIGIT}' — {len(dataset)} images")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

img, label = dataset[0]

plt.title(label)
plt.imshow(img.squeeze(), cmap='gray')
plt.show()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh() # bcz real images are normalized to [-1,1] , fake images should match that range
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *img_shape)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1) #flattening
        return self.model(img)
    
generator = Generator().to(device)
discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

fixed_z = torch.randn(16, latent_dim, device=device)
os.makedirs("vanilla_gan_outputs", exist_ok=True)

def save_generated_images(epoch):
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(fixed_z).detach().cpu()

    grid = torchvision.utils.make_grid(
        gen_imgs,
        nrow=4,
        normalize=True
    )

    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(f"Epoch {epoch}")
    plt.show()

    save_image(
        gen_imgs,
        f"vanilla_gan_outputs/epoch_{epoch}.png",
        nrow=4,
        normalize=True
    )

    generator.train()

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        valid = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        # ---- Train Discriminator ----
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(
            discriminator(real_imgs), valid
        )

        z = torch.randn(batch_size_curr, latent_dim, device=device)
        gen_imgs = generator(z)

        fake_loss = adversarial_loss(
            discriminator(gen_imgs.detach()), fake
        )

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # ---- Train Generator ----
        optimizer_G.zero_grad()

        g_loss = adversarial_loss(
            discriminator(gen_imgs), valid
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

    # ---- Save only at selected epochs ----
    if (epoch + 1) in [1, 50, 100, 500, 1000]:
        save_generated_images(epoch + 1)

# ---- Save model after training ----
os.makedirs("saved_models", exist_ok=True)

torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'epochs': epochs,
}, "saved_models/vanilla_gan.pth")

print("Model saved to saved_models/vanilla_gan.pth")


