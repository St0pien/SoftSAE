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


class SoftTopK(torch.autograd.Function):
    @staticmethod
    def _solve(s, t, a, b, e):
        z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
        ab = torch.where(e > 0, a, b)
        return torch.where(
            e > 0, t + torch.log(z) - torch.log(ab), s - torch.log(z) + torch.log(ab)
        )

    @staticmethod
    def forward(ctx, r, k, alpha, descending=False, high_precision=False):
        assert r.shape[0] == k.shape[0], "k must have same batch size as r"

        # store original dtype and work with float64 when high_precision
        original_dtype = r.dtype
        r_work = r.double() if high_precision else r

        batch_size, num_dim = r_work.shape
        x = torch.empty_like(r_work, requires_grad=False)

        def finding_b():
            scaled = torch.sort(r_work, dim=1)[0]
            scaled.div_(alpha)

            eB = torch.logcumsumexp(scaled, dim=1)
            eB.sub_(scaled).exp_()

            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            eA.add_(scaled).exp_()

            row = torch.arange(1, 2 * num_dim + 1, 2, device=r_work.device)
            torch.add(torch.add(eA, eB, alpha=-1, out=x), row.view(1, -1), out=x)

            w = (k if descending else num_dim - k).unsqueeze(1)
            i = torch.searchsorted(x, 2 * w)
            m = torch.clamp(i - 1, 0, num_dim - 1)
            n = torch.clamp(i, 0, num_dim - 1)

            b = SoftTopK._solve(
                scaled.gather(1, m),
                scaled.gather(1, n),
                torch.where(i < num_dim, eA.gather(1, n), 0),
                torch.where(i > 0, eB.gather(1, m), 0),
                w - i,
            )
            return torch.clamp_max(b, 1e8)

        b = finding_b()

        sign = -1 if descending else 1
        torch.div(r_work, alpha * sign, out=x)
        x.sub_(sign * b)

        sign_x = x > 0
        p = torch.abs(x)
        p.neg_().exp_().mul_(0.5)

        inv_alpha = -sign / alpha
        S = torch.sum(p, dim=1, keepdim=True).mul_(inv_alpha)

        torch.where(sign_x, 1 - p, p, out=p)

        # save float64 tensors but mark the original dtype
        ctx.save_for_backward(r_work, x, S)
        ctx.alpha = alpha
        ctx.original_dtype = original_dtype
        ctx.high_precision = high_precision

        # return in original dtype if high_precision
        return p.to(original_dtype) if high_precision else p

    @staticmethod
    def backward(ctx, grad_output):
        r, x, S = ctx.saved_tensors
        alpha = ctx.alpha
        original_dtype = ctx.original_dtype
        high_precision = ctx.high_precision

        x = x.clone()
        r = r.clone()

        # Work in float64 if high_precision
        grad_output_work = grad_output.double() if high_precision else grad_output

        q_temp = torch.softmax(-torch.abs(x), dim=1)
        qgrad = q_temp * grad_output_work
        grad_k = qgrad.sum(dim=1)
        grad_r = S * q_temp * (grad_k.unsqueeze(1) - grad_output_work)

        # Return in original dtype if high_precision
        return (
            grad_r.to(original_dtype) if high_precision else grad_r,
            None,
            None,
            None,
            None,
        )


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
    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("alpha", torch.tensor(0.001, dtype=torch.float32))

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

    def estimate_k(self, x: torch.Tensor) -> torch.Tensor:
        return (self.k_estimator(x - self.b_dec) * 2 * self.k)[:, 0]

    def encode(self, x: torch.Tensor, return_active: bool = False, use_hard_top_k=True):
        post_relu_feat_acts = F.relu(self.encoder(x - self.b_dec))
        k_estimate = self.estimate_k(x)

        if use_hard_top_k:
            encoded_acts = topk_per_row(post_relu_feat_acts, k_estimate)
        else:
            weights = SoftTopK.apply(
                post_relu_feat_acts, k_estimate, self.alpha, False, True
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

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        self.k_estimator[0].bias.data *= scale

    @classmethod
    def from_pretrained(cls, path, k=None, device=None, **kwargs) -> "SoftTopKSAE":
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = cls(activation_dim, dict_size, k)
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
        dict_class: type = SoftTopKSAE,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        k_anneal_steps: Optional[int] = None,
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
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.k_anneal_steps = k_anneal_steps

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.ae = dict_class(activation_dim, dict_size, k)

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
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = torch.zeros(
            dict_size, dtype=torch.long, device=device
        )
        self.logging_parameters = [
            "effective_l0",
            "dead_features",
            "pre_norm_auxk_loss",
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_fn
        )

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

    def loss(self, x, step=None, logging=False):
        f, active_indices_F, post_relu_acts, estimated_k = self.ae.encode(
            x, return_active=True, use_hard_top_k=False
        )

        x_hat = self.ae.decode(f)

        e = x - x_hat

        self.effective_l0 = self.ae.k.item()

        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts)
        loss = l2_loss + self.auxk_alpha * auxk_loss

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
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)

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
