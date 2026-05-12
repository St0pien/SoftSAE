
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Optional

from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)
from lapsum.topk import soft_topk

def topk_per_row(x, k):
    # x: (B, D)
    # k: (B,)
    B, D = x.shape

    # get max k in batch (scalar)
    k_max = int(k.max())

    # compute only top-k_max
    vals, idx = t.topk(x, k_max, dim=1)

    # build mask for per-row k
    arange = t.arange(k_max, device=x.device)
    mask = arange.unsqueeze(0) < k.unsqueeze(1)  # (B, k_max)

    # zero out values beyond each row's k
    vals = vals * mask

    # scatter back to full tensor
    out = t.zeros_like(x)
    out.scatter_(1, idx, vals)

    return out



class SoftSAEPatched(Dictionary, nn.Module):
    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        alpha: float,
        k_max: Optional[int] = None,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        if k_max is None:
            k_max = k * 2

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        assert 0.0 < alpha < 1, f"alpha={alpha} must be in range: (0.0,1.0)"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("alpha", t.tensor(alpha, dtype=t.float32))
        self.register_buffer("k_max", t.tensor(k_max, dtype=t.int))
        self.register_buffer("norm_factor", t.tensor(1.0, dtype=t.float32))

        self.alpha.requires_grad_(False)

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        k_estimator_encoder = nn.Linear(activation_dim, dict_size)
        k_estimator_encoder.weight.data = self.encoder.weight.data.clone()
        k_estimator_encoder.bias.data.zero_()
        self.k_estimator = nn.Sequential(
            k_estimator_encoder, nn.ReLU(), nn.Linear(dict_size, 1), nn.Sigmoid()
        )

    def estimate_k(self, x: t.Tensor, with_norm_scaling=True):
        # if with_norm_scaling:
        #     x = x / self.norm_factor
        logit = self.k_estimator((x - self.b_dec) / self.norm_factor).squeeze(-1)
        k_hat = logit * self.k_max
        return t.clamp(k_hat, min=1, max=self.dict_size)

    def encode(
        self,
        x: t.Tensor,
        return_active: bool = False,
        use_hard_top_k: bool = True,
        with_norm_scaling=True,
    ):
        # if with_norm_scaling:
        #     x /= self.norm_factor

        post_relu_feat_acts = F.relu(self.encoder(x - self.b_dec))

        if use_hard_top_k:
            with t.no_grad():
                k_estimate = self.estimate_k(x, with_norm_scaling=True).long()
                encoded_acts = topk_per_row(post_relu_feat_acts, k_estimate)
        else:
            k_estimate = self.estimate_k(x, with_norm_scaling=True)
            weights = soft_topk(
                post_relu_feat_acts,
                k_estimate.view(k_estimate.shape[0], 1),
                self.alpha.clone(),
            )
            encoded_acts = post_relu_feat_acts * weights

        if return_active:
            return (
                encoded_acts,
                encoded_acts.sum(0) > 0,
                post_relu_feat_acts,
                k_estimate,
            )
        else:
            return encoded_acts

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(
        self, x: t.Tensor, output_features: bool = False, with_norm_scaling=True
    ):
        encoded_acts_BF, _, _, k_estimate = self.encode(
            x, return_active=True, with_norm_scaling=with_norm_scaling
        )
        x_hat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF, k_estimate

    def scale_biases(self, scale: float):
        self.norm_factor.mul_(scale)

    @classmethod
    def from_pretrained(
        cls, path, k=None, alpha=None, device=None, **kwargs
    ) -> "SoftSAE":
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        if alpha is None:
            alpha = state_dict["alpha"].item()
        elif "alpha" in state_dict and alpha != state_dict["alpha"].item():
            pass
            raise ValueError(
                f"alpha={alpha} != {state_dict['alpha'].item()}=state_dict['alpha']"
            )

        autoencoder = cls(activation_dim, dict_size, k, alpha)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        
        autoencoder.encoder.bias.data *= autoencoder.norm_factor
        autoencoder.b_dec.data *= autoencoder.norm_factor
        return autoencoder