import torch
import torch.nn as nn


class SpectrogramVAE(nn.Module):
    def __init__(self, freq_bins, time_frames, latent_dim=128):
        super().__init__()
        self.freq_bins, self.time_frames = freq_bins, time_frames
        self.latent_dim = latent_dim

        # Encoder conv stack
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()
        )
        # Run dummy through encoder to get shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, freq_bins, time_frames)
            enc_out = self.encoder_conv(dummy)
            _, c_enc, h_enc, w_enc = enc_out.shape
            self._flat_dim = c_enc * h_enc * w_enc
            self._h_enc, self._w_enc = h_enc, w_enc

        # Latent space
        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

        # Decoder fc + deconv stack
        self.fc_dec = nn.Linear(latent_dim, self._flat_dim)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            # IMPORTANT:
            # Output is used to reconstruct spectrogram magnitudes.
            # Magnitudes (or log-magnitudes) are not naturally bounded to [0, 1],
            # therefore sigmoid here can cause training to stall due to saturation.
            # We keep the output linear; the training pipeline defines the target scale.
            nn.ConvTranspose2d(32, 1, 3, 2, 1, output_padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        # Decode
        d = self.fc_dec(z).view(-1, 128, self._h_enc, self._w_enc)
        out = self.decoder_deconv(d)
        # Ensure exact size
        if out.shape[-2:] != (self.freq_bins, self.time_frames):
            out = nn.functional.interpolate(
                out,
                size=(self.freq_bins, self.time_frames),
                mode='bilinear',
                align_corners=False
            )
        return out, mu, logvar
