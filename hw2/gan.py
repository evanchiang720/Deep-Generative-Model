import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

# Prevent WikiArt with PIL error
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder

import ignite
from ignite.engine import Engine, Events
from ignite.metrics import FID, InceptionScore, RunningAverage

try:
    from ignite.handlers import ProgressBar
except ImportError:
    from ignite.contrib.handlers import ProgressBar

# ==========================================
# 1. Settings & Parameters
# ==========================================
data_root = "../public/users/chc302/wikiart"
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

seed = 999
ignite.utils.manual_seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

image_size = 64
batch_size = 128
latent_dim = 100
num_epochs = 30
eval_every = 1
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_workers = 4

# ==========================================
# 2. Dataset & DataLoader
# ==========================================
data_transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class CachedWikiArt(torch.utils.data.Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        img = self.data[idx].float() / 127.5 - 1.0
        return img, 0

train_dataset = CachedWikiArt("/tmp/wikiart_64.pt")
print(f"Total images loaded from cache: {len(train_dataset)}")


# Randomed pick 3000 images for FID/IS
eval_size = min(3000, len(train_dataset))
eval_indices = torch.randperm(len(train_dataset))[:eval_size]
test_dataset = Subset(train_dataset, eval_indices)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
)

# ==========================================
# 3. Model Definition (DCGAN 64x64)
# ==========================================
class Generator3x64x64(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size: 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size: 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size: 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # final state size: 3 x 64 x 64
        )

    def forward(self, x):
        return self.model(x)


class Discriminator3x64x64(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


netG = Generator3x64x64(latent_dim).to(device)
netD = Discriminator3x64x64().to(device)

# ==========================================
# 4. Loss, Fixed noise and Optimizer
# ==========================================
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

# ==========================================
# 5. Ignite Trainging Engine (Trainer)
# ==========================================
real_label = 1.0
fake_label = 0.0


def training_step(engine, data):
    netG.train()
    netD.train()

    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    netD.zero_grad()
    real = data[0].to(device)
    b_size = real.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

    output1 = netD(real).view(-1)
    errD_real = criterion(output1, label)
    errD_real.backward()

    noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
    fake = netG(noise)
    label.fill_(fake_label)

    output2 = netD(fake.detach()).view(-1)
    errD_fake = criterion(output2, label)
    errD_fake.backward()

    errD = errD_real + errD_fake
    optimizerD.step()

    # (2) Update G network: maximize log(D(G(z)))
    netG.zero_grad()
    label.fill_(real_label)

    output3 = netD(fake).view(-1)
    errG = criterion(output3, label)
    errG.backward()
    optimizerG.step()

    return {
        "Loss_G": errG.item(),
        "Loss_D": errD.item(),
        "D_x": output1.mean().item(),
        "D_G_z1": output2.mean().item(),
        "D_G_z2": output3.mean().item(),
    }


trainer = Engine(training_step)


def initialize_fn(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@trainer.on(Events.STARTED)
def init_weights(engine):
    netD.apply(initialize_fn)
    netG.apply(initialize_fn)


G_losses = []
D_losses = []


@trainer.on(Events.ITERATION_COMPLETED)
def store_losses(engine):
    o = engine.state.output
    G_losses.append(o["Loss_G"])
    D_losses.append(o["Loss_D"])


img_list = []


@trainer.on(Events.ITERATION_COMPLETED(every=500))
def store_images(engine):
    netG.eval()
    with torch.no_grad():
        fake = netG(fixed_noise).cpu()
    img_list.append(fake)


# ==========================================
# 6. Ignite Evaluation Engine (FID & Inception Score)
# ==========================================
fid_metric = FID(device=device)
is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])


def interpolate(batch):
    arr = []
    for img in batch:
        # Recover from [-1, 1] to [0, 1], then bilinear interpolate to 299*299
        img_01 = (img.clamp(-1, 1) + 1.0) / 2.0
        pil_img = transforms.ToPILImage()(img_01.cpu())
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


def evaluation_step(engine, batch):
    with torch.no_grad():
        b_size = batch[0].size(0)
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        netG.eval()
        fake_batch = netG(noise)
        fake = interpolate(fake_batch).to(device)
        real = interpolate(batch[0]).to(device)
        return fake, real


evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")

fid_values = []
is_values = []
eval_epochs = []


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    if engine.state.epoch % eval_every != 0:
        return
    evaluator.run(test_dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics["fid"]
    is_score = metrics["is"]
    fid_values.append(fid_score)
    is_values.append(is_score)
    eval_epochs.append(engine.state.epoch)
    print(f"\nEpoch [{engine.state.epoch}/{num_epochs}] Metric Scores")
    print(f"*   FID : {fid_score:.4f} (lower is better)")
    print(f"*    IS : {is_score:.4f} (higher is better)")


# ==========================================
# 7. Loss Recode & Progress Bar
# ==========================================
RunningAverage(output_transform=lambda x: x["Loss_G"]).attach(trainer, "Loss_G")
RunningAverage(output_transform=lambda x: x["Loss_D"]).attach(trainer, "Loss_D")

ProgressBar().attach(trainer, metric_names=["Loss_G", "Loss_D"])
ProgressBar().attach(evaluator)

# ==========================================
# 8. Start Training
# ==========================================
trainer.run(train_dataloader, max_epochs=num_epochs)

# Save Final Weights
torch.save(netG.state_dict(), os.path.join(output_dir, "netG_final.pth"))

# If last interval less than 500 iter, save final images
netG.eval()
with torch.no_grad():
    img_list.append(netG(fixed_noise).cpu())

# ==========================================
# 9. Plot
# ==========================================
# (1) Loss curve
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
plt.show()

# (2) IS & FID curve
fig, ax1 = plt.subplots(figsize=(8, 5))
plt.title("Evaluation Metrics During Training")

color = "tab:red"
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Inception Score (higher is better)", color=color)
ax1.plot(eval_epochs, is_values, marker="o", color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:blue"
ax2.set_ylabel("FID (lower is better)", color=color)
ax2.plot(eval_epochs, fid_values, marker="s", color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.savefig(os.path.join(output_dir, "fid_is_curve.png"), dpi=200)
plt.show()

# (3) Real Images vs Generated Images Comparison
real_batch = next(iter(train_dataloader))

plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Paintings")
plt.imshow(
    np.transpose(
        vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),
        (1, 2, 0),
    )
)

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Generated Paintings (Fake)")
plt.imshow(
    np.transpose(
        vutils.make_grid(img_list[-1][:64], padding=2, normalize=True).cpu(),
        (1, 2, 0),
    )
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "real_vs_fake.png"), dpi=200)
plt.show()

print(f"All figures and model weights saved to {output_dir}/")
print(f"Final FID: {fid_values[-1]:.4f}, Final IS: {is_values[-1]:.4f}")