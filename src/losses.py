import torch
import torch.nn.functional as F

def gan_loss_D(real_out, fake_out):
    # BCE with logits
    real_loss = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
    fake_loss = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
    return (real_loss + fake_loss) * 0.5

def gan_loss_G(fake_out):
    # want discriminator to predict ones for fake
    loss = F.binary_cross_entropy_with_logits(fake_out, torch.ones_like(fake_out))
    return loss

def bce_pixel(pred, target):
    return F.binary_cross_entropy(pred, target)

def kl_divergence(pred, target, eps=1e-8):
    # pred and target are probability maps normalized to sum=1
    p = pred.view(pred.size(0), -1)
    g = target.view(target.size(0), -1)
    p = p / (p.sum(dim=1, keepdim=True) + eps)
    g = g / (g.sum(dim=1, keepdim=True) + eps)
    return (g * (g.log() - p.log())).sum(dim=1).mean()
