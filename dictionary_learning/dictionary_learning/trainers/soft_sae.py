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
    _, D = x.shape

    # sort each row descending
    vals, idx = t.sort(x, dim=1, descending=True)

    # create a mask: True for positions < k[i]
    arange = t.arange(D, device=x.device)
    mask = arange.unsqueeze(0) < k.unsqueeze(1)  # (B, D)

    # zero out values beyond top k[i]
    vals = vals * mask

    # scatter back to original positions
    out = t.zeros_like(x)
    out.scatter_(1, idx, vals)

    return out


class SoftSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, alpha: float):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        assert 0.0 < alpha < 1, f"alpha={alpha} must be in range: (0.0,1.0)"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("alpha", t.tensor(alpha, dtype=t.float32))
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
            k_estimator_encoder, nn.ReLU(), nn.Linear(dict_size, 1)
        )

    def estimate_k(self, x: t.Tensor, with_norm_scaling=True):
        if with_norm_scaling:
            x = x / self.norm_factor
        logit = self.k_estimator(x - self.b_dec)
        k_hat = t.exp(logit).squeeze(-1)
        return t.clamp(k_hat, min=1, max=self.dict_size)

    def encode(
        self,
        x: t.Tensor,
        return_active: bool = False,
        use_hard_top_k: bool = True,
        with_norm_scaling=True,
    ):
        if with_norm_scaling:
            x /= self.norm_factor

        with t.set_grad_enabled(not use_hard_top_k):
            k_estimate = self.estimate_k(x, with_norm_scaling=False)
            post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

            if use_hard_top_k:
                k_estimate = k_estimate.long()
                encoded_acts_BF = topk_per_row(post_relu_feat_acts_BF, k_estimate)
            else:
                weights = soft_topk(
                    post_relu_feat_acts_BF,
                    k_estimate.view(k_estimate.shape[0], 1),
                    self.alpha.clone(),
                )

                encoded_acts_BF = post_relu_feat_acts_BF * weights

        if return_active:
            return (
                encoded_acts_BF,
                encoded_acts_BF.sum(0) > 0,
                post_relu_feat_acts_BF,
                k_estimate,
            )
        else:
            return encoded_acts_BF

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
            raise ValueError(
                f"alpha={alpha} != {state_dict["alpha"].item()}=state_dict['k']"
            )

        autoencoder = cls(activation_dim, dict_size, k, alpha)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class SoftSAETrainer(SAETrainer):
    ae: SoftSAE

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        k_loss_weight=1.0,
        soft_topk_alpha=0.0001,
        dict_class: type = SoftSAE,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        dead_feature_threshold=10_000_000,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        k_anneal_steps: Optional[int] = None,
        alpha_anneal_steps: Optional[int] = None,
        hard_topk_steps: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "SoftSAE",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = f"{wandb_name}_{lm_name}_{activation_dim}_{dict_size}_{k}"
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.k_loss_weight = k_loss_weight
        self.k_anneal_steps = k_anneal_steps
        self.soft_topk_alpha = soft_topk_alpha
        self.alpha_anneal_steps = alpha_anneal_steps
        self.hard_topk_steps = hard_topk_steps
        self.dead_feature_threshold = dead_feature_threshold

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        alpha = soft_topk_alpha if alpha_anneal_steps is None else 0.99
        self.ae = dict_class(activation_dim, dict_size, k, alpha)

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = dead_feature_threshold
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.logging_parameters = [
            "effective_l0",
            "dead_features",
            "pre_norm_auxk_loss",
            "avg_k",
            "min_k",
            "max_k",
            "k_loss",
            "ae_soft_topk_alpha",
            "use_hard_topk",
            "lr_log",
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
        self.avg_k = -1
        self.min_k = -1
        self.max_k = -1
        self.k_loss = -1
        self.ae_soft_topk_alpha = 1
        self.use_hard_topk = 0

        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
    ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None:
            return

        assert (
            0 <= k_anneal_steps < self.steps
        ), "k_anneal_steps must be >= 0 and < steps."
        # self.k is the target k set for the trainer, not the dictionary's current k
        assert activation_dim > self.k, "activation_dim must be greater than k"

        step = min(step, k_anneal_steps)
        ratio = step / k_anneal_steps
        annealed_value = activation_dim * (1 - ratio) + self.k * ratio

        # Update in-place
        self.ae.k.fill_(int(annealed_value))

    def update_annealed_alpha(
        self, step: int, alpha_anneal_steps: Optional[int] = None
    ):
        if alpha_anneal_steps is None or alpha_anneal_steps == 0:
            return

        assert (
            0 <= alpha_anneal_steps < self.steps
        ), "alpha_anneal_steps must be >= 0 and < steps."

        step = min(step, alpha_anneal_steps)
        ratio = step / alpha_anneal_steps
        annealed_value = (1 - ratio) + self.soft_topk_alpha * ratio
        self.ae.alpha.fill_(annealed_value)

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def get_k_loss(self, estimated_k: t.Tensor):
        return t.clamp_min(estimated_k.mean() - self.ae.k, 0)

    def loss(self, x, step=None, logging=False):

        use_hard_topk = self.hard_topk_steps is not None and step > (
            self.steps - self.hard_topk_steps
        )

        f, active_indices_F, post_relu_acts_BF, estimated_k = self.ae.encode(
            x, return_active=True, use_hard_top_k=use_hard_topk
        )

        x_hat = self.ae.decode(f)

        e = x - x_hat

        self.effective_l0 = self.ae.k.item()

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        k_loss = self.get_k_loss(estimated_k) if not use_hard_topk else 0.0
        self.avg_k = estimated_k.mean(dtype=t.float32)
        self.min_k = estimated_k.min()
        self.max_k = estimated_k.max()
        self.k_loss = k_loss
        self.ae_soft_topk_alpha = self.ae.alpha.item()
        self.use_hard_topk = 1 if use_hard_topk else 0
        self.lr_log = self.scheduler.get_last_lr()[0]

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
        loss = l2_loss + self.k_loss_weight * k_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x):
        if step == 0:
            # VERIFY: Should we scale x before median?
            median = self.geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)
        self.update_annealed_alpha(step, self.alpha_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "SoftTopKTrainer",
            "dict_class": "SoftSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "top_k_aux": self.top_k_aux,
            "dead_feature_threshold": self.dead_feature_threshold,
            "k_loss_weight": self.k_loss_weight,
            "k_anneal_steps": self.k_anneal_steps,
            "alpha_anneal_steps": self.alpha_anneal_steps,
            "soft_topk_alpha": self.soft_topk_alpha,
            "hard_topk_steps": self.hard_topk_steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = t.zeros_like(guess)
        weights = t.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break

        return guess
