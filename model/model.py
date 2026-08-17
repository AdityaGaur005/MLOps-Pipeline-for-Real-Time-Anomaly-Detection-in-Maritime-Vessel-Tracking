import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerVAE(nn.Module):
    """
    Encoder: Transformer over the 30-step window -> pooled -> (mu, logvar)
    Decoder: latent z -> repeated across 30 steps -> Transformer -> reconstruction
    """
    def __init__(self, n_features=7, seq_len=30, d_model=64, nhead=4,
                 num_layers=2, latent_dim=16, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        # --- Encoder ---
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # --- Decoder ---
        self.latent_to_seq = nn.Linear(latent_dim, d_model)
        self.pos_dec = PositionalEncoding(d_model, max_len=seq_len)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n_features)

    def encode(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h = h.mean(dim=1)  # pool over time
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.latent_to_seq(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        h = self.pos_dec(h)
        h = self.decoder(h)
        return self.output_proj(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, kl_weight=0.01):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='none').mean(dim=[1, 2])  # per-sample
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / x.size(1)
    total = recon_loss + kl_weight * kl
    return total.mean(), recon_loss.mean(), kl.mean()
