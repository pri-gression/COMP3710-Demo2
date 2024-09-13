import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths to OASIS data (adjust according to your environment)
data_path = '/home/groups/comp3710/OAI'

# Preprocessing: Normalize images to [-1, 1] and resize to 64x64
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

# Load data
train_data = datasets.ImageFolder(root=data_path, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64 * 64 * 3),  # 64 * 64 * 3 for RGB output
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 64, 64)  # Reshape into 3-channel 64x64 images (RGB)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64 * 64 * 3, 512),  # Input size matches 64x64 RGB image flattened (64*64*3 = 12288)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a probability (real or fake)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # Flatten the image (batch_size, 12288 for RGB)
        validity = self.model(img_flat)
        return validity

# Hyperparameters
latent_dim = 128
epochs =  20
learning_rate = 0.0001
show_images_interval = 20  # Only show images every 'n' epochs

# Initialize generator and discriminator
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Function to save generated images during training
def save_generated_images(epoch, fixed_noise):
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    img_grid = vutils.make_grid(fake, padding=2, normalize=True)
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.title(f"Generated Images at Epoch {epoch}")
    plt.savefig(f"/home/Student/s4760436/generated_images/generated_images_epoch_{epoch}.png")

# Training Loop
fixed_noise = torch.randn(25, latent_dim).to(device)  # For generating consistent images during training

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        
        # Real and fake labels
        real = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)
        
        # Train Discriminator
        optimizer_D.zero_grad()

        real_imgs = imgs.to(device)
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)

        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        gen_loss = adversarial_loss(discriminator(gen_imgs), real)

        gen_loss.backward()
        optimizer_G.step()

    # Only print loss and save images every 'show_images_interval' epochs
    if (epoch + 1) % show_images_interval == 0 or epoch == epochs - 1:
        print(f"Epoch [{epoch + 1}/{epochs}] D_loss: {d_loss.item()}, G_loss: {gen_loss.item()}")
        save_generated_images(epoch + 1, fixed_noise)

print("Training finished!")

def plot_losses(g_losses, d_losses, output_dir="output/"):
    os.makedirs(output_dir, exist_ok=True) 
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_plot.png")) 
    plt.close() 
