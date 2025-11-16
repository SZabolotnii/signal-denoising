import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv_block(in_ch, out_ch, k=4, s=2, p=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True)
    )


class UnetAutoencoder(nn.Module):
    """
    Глибший U-Net для прогнозу маски на спектрограмі:
    Вхід: (B,1,F,T) — нормалізований log1p(|STFT|).
    Вихід: (B,1,F,T) — mask ∈ [0,1].
    """
    def __init__(self, input_shape):
        super().__init__()
        # Encoder (даунсемпл по F і T)
        self.enc1 = nn.Sequential(
            conv_block(1, 32), conv_block(32, 32)
        )
        self.down1 = conv_block(32, 64, k=4, s=2, p=1)   # /2
        self.enc2 = nn.Sequential(
            conv_block(64, 64), conv_block(64, 64)
        )
        self.down2 = conv_block(64, 128, k=4, s=2, p=1)  # /4
        self.enc3 = nn.Sequential(
            conv_block(128, 128), conv_block(128, 128)
        )
        self.down3 = conv_block(128, 256, k=4, s=2, p=1) # /8

        self.bottleneck = nn.Sequential(
            conv_block(256, 256), conv_block(256, 256)
        )

        # Decoder
        self.up3 = deconv_block(256, 128)  # x2
        self.dec3 = nn.Sequential(
            conv_block(256, 128), conv_block(128, 128)
        )
        self.up2 = deconv_block(128, 64)   # x4
        self.dec2 = nn.Sequential(
            conv_block(128, 64), conv_block(64, 64)
        )
        self.up1 = deconv_block(64, 32)    # x8
        self.dec1 = nn.Sequential(
            conv_block(64, 32), conv_block(32, 32)
        )

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.target_shape = input_shape  # (F,T)

    @staticmethod
    def _crop_to(x, ref):
        _, _, h, w = x.shape
        _, _, H, W = ref.shape
        dh = h - H
        dw = w - W
        if dh == 0 and dw == 0:
            return x
        return x[..., dh // 2:h - (dh - dh // 2), dw // 2:w - (dw - dw // 2)]

    def forward(self, x):
        # x: (B,1,F,T)
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        b = self.bottleneck(d3)

        u3 = self.up3(b)
        e3c = self._crop_to(e3, u3)
        u3 = torch.cat([u3, e3c], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        e2c = self._crop_to(e2, u2)
        u2 = torch.cat([u2, e2c], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        e1c = self._crop_to(e1, u1)
        u1 = torch.cat([u1, e1c], dim=1)
        u1 = self.dec1(u1)

        out = self.out_conv(u1)
        out = self.sigmoid(out)  # mask ∈ [0,1]

        # підгін до точної (F,T), якщо треба
        if out.shape[-2:] != self.target_shape:
            out = F.interpolate(out, size=self.target_shape, mode='bilinear', align_corners=False)
        return out
