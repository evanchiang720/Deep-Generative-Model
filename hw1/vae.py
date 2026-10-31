import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

import PIL.Image as Image
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

channels_encoder = [1, 32, 64, 128]
channels_decoder = [128, 64, 32, 1]

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()

        layers = []

        # encoder
        for i in range(len(channels_encoder) - 1):
            layers += [nn.Conv2d(channels_encoder[i], channels_encoder[i+1], kernel_size=4, stride=2, padding=1),
                       nn.BatchNorm2d(channels_encoder[i+1]),
                       nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(channels_encoder[-1] * 64, latent_dim)
        self.fc_log_var = nn.Linear(channels_encoder[-1] * 64, latent_dim)

        # decoder
        self.decoder_input = nn.Linear(latent_dim, channels_encoder[-1] * 64)
        layers = []
        for i in range(len(channels_decoder) - 2):
            layers += [nn.ConvTranspose2d(channels_decoder[i], channels_decoder[i+1], kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(channels_decoder[i+1]),
                        nn.ReLU(inplace=True)]
        layers += [nn.ConvTranspose2d(channels_decoder[-2], channels_decoder[-1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(channels_decoder[-1]),
                    nn.Sigmoid()]
        self.decoder = nn.Sequential(*layers)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = std * epsilon + mu
        return z

    def forward(self, x):
        latent = self.encoder(x)
        latent = torch.flatten(latent, start_dim=1)
        mu = self.fc_mu(latent)
        log_var = self.fc_log_var(latent)
        z = self.reparameterize(mu, log_var)
        result = self.decoder_input(z)
        result = result.view(-1, 128, 8, 8)
        recon_x = self.decoder(result)
        return mu, log_var, recon_x


def vae_loss(x, recon_x, mu, log_var):
    kld_loss = -torch.mean(0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    recon_loss = nn.functional.mse_loss(x, recon_x, reduction="mean")
    return  0.001 * kld_loss +recon_loss

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)

dataset = ImageFolder(root="chest_xray/train", transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

latent_dim = 256

model = VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []

# train
num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss = 0
    for images, _ in dataloader:
        images = images.to(device)
        mu, log_var, recon_x = model(images)
        loss = vae_loss(images, recon_x, mu, log_var)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(epoch_loss / len(dataloader))

# draw loss curve
plt.plot(train_losses, marker='.', color='blue')
plt.title("Training Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("training_loss.png", bbox_inches="tight", dpi=300)
plt.show()

# test
model.eval()
with torch.no_grad():
    z = torch.randn(36, latent_dim, device=device)
    result = model.decoder_input(z)
    result = result.view(-1, 128, 8, 8)
    generated_images = model.decoder(result).cpu()

# output
fig, axes = plt.subplots(6, 6, figsize=(10, 10))
axes = axes.flatten()
for i in range(36):
    img_data = generated_images[i]
    img_data = img_data.squeeze()
    axes[i].imshow(img_data.squeeze(), cmap='gray') 
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("generated_x_ray.png", bbox_inches="tight", dpi=300)
plt.show()

# evaluation
fid_metric = FID(device=device)
is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])

def interpolate(batch):
    arr = []
    for img in batch:
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        pil_img = transforms.ToPILImage()(img.cpu())
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr).to(device)

def evaluation_step(engine, batch):
    with torch.no_grad():
        batch_size = batch[0].size(0)
        noise = torch.randn(batch_size, latent_dim, device=device)
        model.eval()
        result = model.decoder_input(noise)
        result = result.view(-1, 128, 8, 8)
        result = result.to(device)
        fake_batch = model.decoder(result)
        fake = interpolate(fake_batch)
        real = interpolate(batch[0])
        return fake, real

evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")

evaluator.run(dataloader, max_epochs=1)
metrics = evaluator.state.metrics
fid_score = metrics['fid']
is_score = metrics['is']
print(f"*   FID : {fid_score:4f}")
print(f"*    IS : {is_score:4f}")