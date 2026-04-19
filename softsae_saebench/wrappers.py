from dictionary_learning.trainers.soft_sae import SoftSAE
from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig
import torch


class SAEBenchSoftSAE:
    def __init__(self, sae: SoftSAE):
        self.sae = sae
        self.W_dec = self.sae.decoder.weight.T
        self.W_enc = self.sae.encoder.weight.T
        self.dtype = self.W_dec.dtype
        self.device = self.W_dec.device
        self.cfg = CustomSAEConfig(
            "google/gemma-2-2b",
            self.W_dec.shape[1],
            self.W_dec.shape[0],
            12,
            "resid_post_layer",
        )

    def encode(self, x: torch.Tensor):
        return self.sae.encode(x)

    def decode(self, x: torch.Tensor):
        return self.decode(x)

    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))

    def to(self, *args, **kwargs):
        self.sae.to(*args, **kwargs)
        self.device = self.W_dec.device
        self.dtype = self.W_dec.dtype

        return self
