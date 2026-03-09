# models/autoencoder_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- helpers --------
def conv_block(cin, cout, k=3, s=1, p=1):
    """2x(Conv+BN+LeakyReLU).  # CHANGED: додано нормалізацію та подвійні Conv"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(cout, cout, k, 1, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(inplace=True),
    )

class SE(nn.Module):
    """Squeeze-and-Excitation для channel attention.  # NEW"""
    def __init__(self, c, r=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)

# -------- model --------
class UnetAutoencoder(nn.Module):
    """
    Mask U-Net: замість чистої амплітуди модель передбачає МАСКУ ∈[0,1] для приглушення шуму.
    # CHANGED: архітектура стала глибшою, даунсемпл по F та T, skip-зв’язки, SE-attention.
    """
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape  # (F, T)

        # Encoder (3 рівні)
        self.e1 = conv_block(1,   32);  self.se1 = SE(32)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)                  # ↓F, ↓T  # CHANGED

        self.e2 = conv_block(32,  64);  self.se2 = SE(64)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)                  # ↓F, ↓T  # CHANGED

        self.e3 = conv_block(64, 128);  self.se3 = SE(128)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)                  # ↓F, ↓T  # CHANGED

        # Bottleneck
        self.bott = conv_block(128, 256); self.seb = SE(256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # ↑F, ↑T  # CHANGED
        self.d3  = conv_block(128 + 128, 128); self.sed3 = SE(128)

        self.up2 = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2) # ↑F, ↑T
        self.d2  = conv_block(64 + 64,   64);  self.sed2 = SE(64)

        self.up1 = nn.ConvTranspose2d( 64,  32, kernel_size=2, stride=2) # ↑F, ↑T
        self.d1  = conv_block(32 + 32,   32);  self.sed1 = SE(32)

        # Вихід: маска ∈ [0,1]
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        # Масштабування до цільової форми, якщо трохи не співпало після апсемплу
        self._final_interp = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def _crop_to_match(enc_feat, target_feat):
        """Центральний кроп, якщо просторові розміри не співпали."""
        _, _, h, w = enc_feat.shape
        _, _, th, tw = target_feat.shape
        dh, dw = h - th, w - tw
        if dh == 0 and dw == 0:
            return enc_feat
        enc_feat = enc_feat[..., dh // 2:h - (dh - dh // 2), dw // 2:w - (dw - dw // 2)]
        return enc_feat

    def forward(self, x):
        """
        x: (B,1,F,T) — амплітуда спектрограми NOISY (а не clean!)
        return: (B,1,F,T) — маска для амплітуди
        """
        # Encoder
        e1 = self.se1(self.e1(x))      # (B,32,F,T)
        p1 = self.p1(e1)               # (B,32,F/2,T/2)

        e2 = self.se2(self.e2(p1))     # (B,64,F/2,T/2)
        p2 = self.p2(e2)               # (B,64,F/4,T/4)

        e3 = self.se3(self.e3(p2))     # (B,128,F/4,T/4)
        p3 = self.p3(e3)               # (B,128,F/8,T/8)

        # Bottleneck
        b  = self.seb(self.bott(p3))   # (B,256,F/8,T/8)

        # Decoder
        u3 = self.up3(b)               # (B,128,F/4,T/4)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = self._final_interp(u3, e3.shape[-2:])
        d3 = self.sed3(self.d3(torch.cat([u3, e3], dim=1)))  # (B,128,F/4,T/4)

        u2 = self.up2(d3)              # (B,64,F/2,T/2)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = self._final_interp(u2, e2.shape[-2:])
        d2 = self.sed2(self.d2(torch.cat([u2, e2], dim=1)))  # (B,64,F/2,T/2)

        u1 = self.up1(d2)              # (B,32,F,T)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = self._final_interp(u1, e1.shape[-2:])
        d1 = self.sed1(self.d1(torch.cat([u1, e1], dim=1)))  # (B,32,F,T)

        mask = torch.sigmoid(self.out_conv(d1))              # (B,1,F,T)  # CHANGED: mask ∈ [0,1]
        if mask.shape[-2:] != self.input_shape:
            mask = self._final_interp(mask, self.input_shape)
        return mask
