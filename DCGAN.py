import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dim = 100
batch_size = 128
epochs = 500
lr = 0.0002
img_channels = 1

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST(
    root="/kaggle/working",
    train=False,
    transform=transform,
    download=True
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # Input: Z → (batch, 100, 1, 1)
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False), # output 7x7 image
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # output 14x14 image
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, img_channels, 4, 2, 1, bias=False), #output 28x28 image
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        valid = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        real_loss = criterion(discriminator(real_imgs), valid)

        z = torch.randn(batch_size_curr, latent_dim, 1, 1, device=device)
        gen_imgs = generator(z)

        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        if i % 300 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{i}/{len(dataloader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )

os.makedirs("saved_models", exist_ok=True)

torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'epochs': epochs,
}, "saved_models/dcgan.pth")

print("Model saved to saved_models/dcgan.pth")
