from collections import namedtuple
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.trainer import (
    get_lr_schedule,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)
from dictionary_learning.trainers.trainer import SAETrainer
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch

from lapsum.topk import soft_topk


def topk_per_row(x, k):
    # x: (B, D)
    # k: (B,)
    _, D = x.shape

    # sort each row descending
    vals, idx = torch.sort(x, dim=1, descending=True)

    # create a mask: True for positions < k[i]
    arange = torch.arange(D, device=x.device)
    mask = arange.unsqueeze(0) < k.unsqueeze(1)  # (B, D)

    # zero out values beyond top k[i]
    vals = vals * mask

    # scatter back to original positions
    out = torch.zeros_like(x)
    out.scatter_(1, idx, vals)

    return out


class SoftTopKSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int, alpha: float):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("norm_factor", torch.tensor(1.0))
        self.register_buffer(
            "shift_factor", torch.zeros(activation_dim, dtype=torch.float32)
        )

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

        k_estimator_encoder = nn.Linear(activation_dim, dict_size)
        k_estimator_encoder.weight.data = self.encoder.weight.data.clone()
        k_estimator_encoder.bias.data.zero_()
        self.k_estimator = nn.Sequential(
            k_estimator_encoder, nn.ReLU(), nn.Linear(dict_size, 1), nn.Sigmoid()
        )

    def normalize(self, x: torch.Tensor):
        return (x - self.shift_factor) / self.norm_factor

    def denormalize(self, x: torch.Tensor):
        return x * self.norm_factor + self.shift_factor

    def estimate_k(self, x: torch.Tensor) -> torch.Tensor:
        return (self.k_estimator(x - self.b_dec) * 2 * self.k)[:, 0]

    def encode(self, x: torch.Tensor, return_active: bool = False, use_hard_topk=True):
        post_relu_feat_acts = F.relu(self.encoder(x - self.b_dec))
        k_estimate = self.estimate_k(x)

        if use_hard_topk:
            encoded_acts = topk_per_row(post_relu_feat_acts, k_estimate)
        else:
            weights = soft_topk(
                post_relu_feat_acts,
                k_estimate.view((k_estimate.shape[0], 1)),
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

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts, _, _, k_estimate = self.encode(x, return_active=True)
        x_hat = self.decode(encoded_acts)

        if not output_features:
            return x_hat
        else:
            return x_hat, encoded_acts, k_estimate

    def set_normalization(self, norm_factor: float, shift: torch.Tensor):
        self.norm_factor.fill_(norm_factor)
        self.shift_factor.copy_(shift)

    @classmethod
    def from_pretrained(
        cls, path, k=None, alpha=None, device=None, **kwargs
    ) -> "SoftTopKSAE":
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        if alpha is None:
            k = state_dict["alpha"].item()
        elif "alpha" in state_dict and alpha != state_dict["alpha"].item():
            raise ValueError(
                f"alpha={k} != {state_dict['alpha'].item()}=state_dict['alpha']"
            )

        autoencoder = cls(activation_dim, dict_size, k, alpha)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class SoftTopKTrainer(SAETrainer):
    ae: SoftTopKSAE

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        k_loss_type="budget",
        k_loss_weight=1.0,
        soft_topk_alpha=0.0001,
        dict_class: type = SoftTopKSAE,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        dead_feature_threshold=500_000,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        k_anneal_steps: Optional[int] = None,
        alpha_anneal_steps: Optional[int] = None,
        hard_topk_steps: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "SoftTopKSAE",
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
        self.k_anneal_steps = k_anneal_steps
        self.k_loss_type = k_loss_type
        self.k_loss_weight = k_loss_weight
        self.soft_topk_alpha = soft_topk_alpha
        self.alpha_anneal_steps = alpha_anneal_steps
        self.hard_topk_steps = hard_topk_steps

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.ae = dict_class(
            activation_dim,
            dict_size,
            activation_dim if k_anneal_steps is not None else k,
            1 if alpha_anneal_steps is not None else soft_topk_alpha,
        )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        self.num_tokens_since_fired = torch.zeros(
            dict_size, dtype=torch.long, device=device
        )
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
            "lr",
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

        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_fn
        )

        self.loss_map = {"budget": self.get_budget_loss, "kl0": self.get_kl0_loss}

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
    ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None or k_anneal_steps == 0:
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

    def get_auxiliary_loss(
        self, residual_BD: torch.Tensor, post_relu_acts_BF: torch.Tensor
    ):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            auxk_latents = torch.where(
                dead_features[None], post_relu_acts_BF, -torch.inf
            )

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = torch.zeros_like(post_relu_acts_BF)
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
            return torch.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def get_kl0_loss(self, estimated_k: torch.Tensor):
        return estimated_k.mean() / self.ae.k

    def get_budget_loss(self, estimated_k: torch.Tensor):
        return torch.clamp_min(estimated_k.mean() - self.ae.k, 0)

    def loss(self, x, step=None, logging=False):
        x_norm = self.ae.normalize(x)

        use_hard_topk = False
        if self.hard_topk_steps is not None and step > (
            self.steps - self.hard_topk_steps
        ):
            use_hard_topk = True

        f, active_indices_F, post_relu_acts, estimated_k = self.ae.encode(
            x_norm, return_active=True, use_hard_topk=use_hard_topk
        )

        x_hat = self.ae.decode(f)

        e = x_norm - x_hat

        self.effective_l0 = self.ae.k.item()

        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        k_loss = self.loss_map[self.k_loss_type](estimated_k)
        self.avg_k = estimated_k.mean()
        self.min_k = estimated_k.min()
        self.max_k = estimated_k.max()
        self.k_loss = k_loss
        self.ae_soft_topk_alpha = self.ae.alpha.item()
        self.use_hard_topk = 1 if use_hard_topk else 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts)
        loss = l2_loss + self.k_loss_weight * k_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x_norm,
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
            median = self.geometric_median(self.ae.normalize(x))
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
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

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
            "dict_class": "SoftTopKSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "top_k_aux": self.top_k_aux,
            "dead_feature_threshold": self.dead_feature_threshold,
            "k_loss_type": self.k_loss_type,
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
    def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = torch.zeros_like(guess)
        weights = torch.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / torch.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if torch.norm(guess - prev) < tol:
                break

        return guess
