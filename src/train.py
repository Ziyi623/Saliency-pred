import torch
import os
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from dataset import get_loaders
from losses import gan_loss_D, gan_loss_G
import torch.optim as optim
from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter
from config import cfg


def train():
    # ---------------------------
    # Data loader
    # ---------------------------
    loader = get_loaders(
        cfg.train_stimuli_dir,
        cfg.train_gt_dir,
        batch_size=cfg.batch_size,
        img_size=(cfg.img_height, cfg.img_width)
    )
    device = cfg.device
    # ---------------------------
    # Models
    # ---------------------------
    G = UNetGenerator().to(device)
    D = PatchDiscriminator(in_channels=4).to(device)

    # ---------------------------
    # Optimizers
    # ---------------------------
    optG = optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    optD = optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)

    # L1 loss (saliency regression)
    l1_loss = L1Loss()

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(cfg.epochs):
        G.train()
        D.train()

        for step, (imgs, maps) in enumerate(loader):
            imgs = imgs.to(device)          # RGB image
            maps = maps.to(device)          # GT saliency map

            # ==============================
            # 1. Train Discriminator
            # ==============================
            with torch.no_grad():
                fake_maps = G(imgs)  # generated saliency maps

            real_out = D(imgs, maps)
            fake_out = D(imgs, fake_maps.detach())
            lossD = gan_loss_D(real_out, fake_out)

            optD.zero_grad()
            lossD.backward()
            optD.step()

            # ==============================
            # 2. Train Generator
            # ==============================
            fake_maps = G(imgs)
            fake_out = D(imgs, fake_maps)

            loss_adv = gan_loss_G(fake_out)
            loss_l1 = l1_loss(fake_maps, maps)

            lossG = cfg.lambda_adv * loss_adv + cfg.lambda_l1 * loss_l1

            optG.zero_grad()
            lossG.backward()
            optG.step()

            # --------------------------
            # Logging
            # --------------------------
            if step % 20 == 0:
                print(f"Epoch [{epoch}/{cfg.epochs}] Step [{step}]  "
                      f"lossD: {lossD.item():.4f}  lossG: {lossG.item():.4f}  "
                      f"L1: {loss_l1.item():.4f}  Adv: {loss_adv.item():.4f}")

                writer.add_scalar("Loss/D", lossD.item(), epoch * len(loader) + step)
                writer.add_scalar("Loss/G", lossG.item(), epoch * len(loader) + step)
                writer.add_scalar("Loss/L1", loss_l1.item(), epoch * len(loader) + step)
                writer.add_scalar("Loss/Adv", loss_adv.item(), epoch * len(loader) + step)

        # --------------------------
        # Save model every epoch
        # --------------------------
        torch.save(G.state_dict(), os.path.join(cfg.model_save_dir, f"G_epoch{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(cfg.model_save_dir, f"D_epoch{epoch}.pth"))

    print("Training Finished!")


if __name__ == "__main__":
    train()
