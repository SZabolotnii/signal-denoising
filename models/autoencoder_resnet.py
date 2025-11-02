import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        # Encoder (smaller)
        self.enc1 = ResidualBlock(1, 16, downsample=False)
        self.enc2 = ResidualBlock(16, 32, downsample=True)
        self.enc3 = ResidualBlock(32, 64, downsample=True)

        # Decoder (upsample to original shape)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(32, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(16, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)   # [B, 16, H, W]
        e2 = self.enc2(e1)  # [B, 32, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 64, H/4, W/4]

        d1 = self.up1(e3)   # [B, 32, H/2, W/2]
        d1 = self.dec1(d1)
        d2 = self.up2(d1)   # [B, 16, H, W]
        d2 = self.dec2(d2)
        out = self.final(d2)  # [B, 1, H, W]
        out = self.sigmoid(out)
        if out.shape[-2:] != self.input_shape:
            out = nn.functional.interpolate(out, size=self.input_shape, mode='bilinear', align_corners=False)
        return out
