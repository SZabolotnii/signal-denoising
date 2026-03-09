"""
HybridDSGE_UNet — U-Net автоенкодер з DSGE-каналами на вході.

Архітектура ідентична поточному UnetAutoencoder (models/autoencoder_unet.py),
єдина зміна — перший Conv2d приймає (1 + S) каналів замість 1:
    - Канал 0: |STFT(x̃)|          — стандартна амплітудна спектрограма
    - Канали 1..S: |STFT(φᵢ(x̃))| — DSGE-спектрограми від базисних функцій

Вхід: (B, 1+S, F, T') — конкатенація по каналах.
Вихід: (B, 1, F, T') — маска знешумлення (sigmoid → [0, 1]).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridDSGE_UNet(nn.Module):
    """
    U-Net автоенкодер з підтримкою DSGE-каналів.

    Parameters
    ----------
    input_shape : tuple[int, int]
        (F, T') — розміри частотно-часової спектрограми.
    dsge_order : int
        S — кількість DSGE-каналів (порядок апроксимації).
        in_channels = 1 + dsge_order.
    """

    def __init__(self, input_shape: tuple[int, int], dsge_order: int = 3):
        super().__init__()
        self.input_shape = input_shape
        self.dsge_order = dsge_order
        in_channels = 1 + dsge_order  # 1 STFT + S DSGE-каналів

        # ── Encoder ──────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(1, 2), padding=1),  # downsample in time
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──────────────────────────────────────
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.ReLU(inplace=True),
        )
        # Skip-connection: cat([dec1_out, enc1_out]) → 32 канали
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _crop_to_match(enc_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        """Центральне кропання enc_feat до розміру target_feat (по просторових осях)."""
        _, _, h, w = enc_feat.shape
        _, _, th, tw = target_feat.shape
        dh, dw = h - th, w - tw
        return enc_feat[
            ...,
            dh // 2: h - (dh - dh // 2),
            dw // 2: w - (dw - dw // 2),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1+S, F, T')
            Конкатенований вхід: STFT-спектрограма + S DSGE-спектрограм.

        Returns
        -------
        mask : torch.Tensor, shape (B, 1, F, T')
            Маска знешумлення у діапазоні [0, 1].
        """
        e1 = self.enc1(x)   # (B, 16, F, T')
        e2 = self.enc2(e1)  # (B, 32, F, T'//2)

        d1 = self.dec1(e2)  # (B, 16, F, T')
        if d1.shape[-2:] != e1.shape[-2:]:
            e1 = self._crop_to_match(e1, d1)
        d1 = torch.cat([d1, e1], dim=1)  # (B, 32, F, T')

        out = self.final_conv(d1)         # (B, 1, F, T')
        out = self.sigmoid(out)

        if out.shape[-2:] != self.input_shape:
            out = F.interpolate(out, size=self.input_shape, mode='bilinear', align_corners=False)

        return out

    def param_count(self) -> int:
        """Повертає кількість параметрів моделі."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────
#  Швидка перевірка форм
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import torch

    F_bins, T_frames = 65, 54       # типовий розмір для nperseg=128, noverlap=96, T=2144
    S = 3
    B = 4

    model = HybridDSGE_UNet(input_shape=(F_bins, T_frames), dsge_order=S)
    x = torch.randn(B, 1 + S, F_bins, T_frames)
    out = model(x)

    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {model.param_count():,}")
    assert out.shape == (B, 1, F_bins, T_frames), "Shape mismatch!"
    print("✅ Shape OK")
