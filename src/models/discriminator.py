import torch.nn as nn
import torch

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4):  # concat rgb + saliency(1)
        super().__init__()
        def conv(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.net = nn.Sequential(
            conv(in_channels, 64, 2),
            conv(64, 128, 2),
            conv(128, 256, 2),
            conv(256, 512, 1),
            nn.Conv2d(512, 1, 4, padding=1)  # output patch map
        )

    def forward(self, img, sal_map):
        # img: B x 3 x H x W, sal_map: B x 1 x H x W
        x = torch.cat([img, sal_map], dim=1)
        return self.net(x)  # not sigmoid, use BCEWithLogitsLoss
