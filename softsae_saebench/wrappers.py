from dictionary_learning.trainers.soft_sae import SoftSAE
from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig
import torch


class SAEBenchSAE:
    def __init__(self, sae):
        self.sae = sae
        if not hasattr(self.sae, "W_dec"):
            self.W_dec = self.sae.decoder.weight.T
            self.W_enc = self.sae.encoder.weight.T
        else:
            self.W_dec = self.sae.W_dec
            self.W_enc = self.sae.W_enc
        self.dtype = self.W_dec.dtype
        self.device = self.W_dec.device
        self.cfg = CustomSAEConfig(
            "google/gemma-2-2b",
            self.W_dec.shape[1],
            self.W_dec.shape[0],
            12,
            "blocks.12.hook_resid_post",
            dtype=self.dtype
        )

    def encode(self, x: torch.Tensor):
        if x.ndim == 3:  # (B, L, F)
            B, L, F = x.shape
            x_flat = x.reshape(B * L, F)
            enc_flat = self.sae.encode(x_flat)
            enc = enc_flat.reshape(B, L, -1)
        elif x.ndim == 2:  # (B, F)
            enc = self.sae.encode(x)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        return enc

    def decode(self, x: torch.Tensor):
        if x.ndim == 3:  # (B, L, H)
            B, L, H = x.shape
            x_flat = x.reshape(B * L, H)
            dec_flat = self.sae.decode(x_flat)
            dec = dec_flat.reshape(B, L, -1)
        elif x.ndim == 2:  # (B, H)
            dec = self.sae.decode(x)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        return dec

    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))

    def to(self, *args, **kwargs):
        self.sae = self.sae.to(*args, **kwargs)
        self.W_dec = self.W_dec.to(*args, **kwargs)
        self.W_enc = self.W_enc.to(*args, **kwargs)

        self.device = self.W_dec.device
        self.dtype = self.W_dec.dtype
        self.cfg.dtype = self.dtype

        return self
