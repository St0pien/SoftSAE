"""
Training dictionary
"""

from enum import Enum
import json
import torch.multiprocessing as mp
import os
from queue import Empty
from typing import Optional, Tuple
from contextlib import nullcontext
from datetime import datetime

import torch as t
from tqdm import tqdm

import wandb


def new_wandb_process(log_queue, run):
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            run.log(log)
        except Empty:
            continue


def log_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list = [],
    verbose: bool = False,
):
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act = z.clone()
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            if not transcoder:
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance explained
                total_variance = t.var(act, dim=0).sum()
                residual_variance = t.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            if verbose:
                print(
                    f"Step {step}: L0 = {l0}, frac_variance_explained = {frac_variance_explained}"
                )

            # log parameters from training
            log.update(
                {
                    f"{k}": v.cpu().item() if isinstance(v, t.Tensor) else v
                    for k, v in losslog.items()
                }
            )
            log[f"l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, t.Tensor):
                    value = value.cpu().item()
                log[f"{name}"] = value

            if log_queues:
                log_queues[i].put(log)


def get_norm_shift_factor(data, steps: int, activation_dim: int) -> float:
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147

    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
    total_mean_squared_norm = 0
    center = t.zeros(activation_dim, dtype=t.float32)
    count = 0

    for step, act_BD in enumerate(
        tqdm(data, total=steps, desc="Calculating norm factor")
    ):
        if step > steps:
            break

        count += 1
        mean_squared_norm = t.mean(t.sum(act_BD**2, dim=1))
        total_mean_squared_norm += mean_squared_norm

        center += t.mean(act_BD, dim=0).cpu()

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = t.sqrt(average_mean_squared_norm).item()

    shift_factor = center / count

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")
    print(f"Shift factor mean: {shift_factor.mean().item()}")

    return norm_factor, shift_factor


class ActivationsNormalization(Enum):
    NONE = 1
    SCALE = 2
    SCALE_SHIFT = 3


def trainSAE(
    data,
    trainer_config: dict,
    steps: int,
    use_wandb: bool = False,
    run: Optional[wandb.Run] = None,
    save_steps: Optional[list[int]] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    activations_split_by_head: bool = False,
    transcoder: bool = False,
    normalize_activations: ActivationsNormalization = ActivationsNormalization.NONE,
    verbose: bool = False,
    device: str = "cuda",
    autocast_dtype: t.dtype = t.float32,
    backup_steps: Optional[int] = None,
) -> str | Tuple[str, wandb.Run]:

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = (
        nullcontext()
        if device_type == "cpu"
        else t.autocast(device_type=device_type, dtype=autocast_dtype)
    )

    config = trainer_config.copy()
    if "wandb_name" in config:
        config["wandb_name"] = f"{config['wandb_name']}"
    trainer_class = config["trainer"]
    del config["trainer"]
    trainer = trainer_class(**config)

    wandb_process = None
    log_queue = None

    if use_wandb:
        log_queue = mp.Queue()
        wandb_process = mp.Process(
            target=new_wandb_process,
            args=(log_queue, run),
        )
        wandb_process.start()

    if save_dir is not None:
        save_dir = os.path.join(save_dir, trainer.config["wandb_name"])
        os.makedirs(save_dir, exist_ok=True)
        config_to_save = {"trainer": trainer.config}
        try:
            config_to_save["buffer"] = data.config
        except:
            pass
        with open(
            os.path.join(save_dir, f"config_{trainer.config['wandb_name']}.json"),
            "w",
        ) as f:
            json.dump(config_to_save, f, indent=4)
    else:
        save_dir = None

    norm_factor, shift_factor = get_norm_shift_factor(
        data, steps=100, activation_dim=trainer.ae.activation_dim
    )

    if normalize_activations != ActivationsNormalization.SCALE_SHIFT:
        shift_factor = 0.0
    if normalize_activations == ActivationsNormalization.NONE:
        norm_factor = 1.0

    trainer.config["norm_factor"] = norm_factor
    if normalize_activations == ActivationsNormalization.SCALE_SHIFT:
        trainer.config["shift_factor_mean"] = shift_factor.mean().item()
    trainer.ae.set_normalization(norm_factor, shift_factor)

    for step, act in enumerate(tqdm(data, total=steps)):
        act = act.to(dtype=autocast_dtype)

        if step >= steps:
            break

        if (use_wandb or verbose) and step % log_steps == 0:
            log_stats(
                [trainer],
                step,
                act,
                activations_split_by_head,
                transcoder,
                log_queues=[log_queue] if use_wandb else [],
                verbose=verbose,
            )

        if save_steps is not None and step in save_steps:
            if save_dir is not None:
                if not os.path.exists(os.path.join(save_dir, "checkpoints")):
                    os.mkdir(os.path.join(save_dir, "checkpoints"))

                checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                t.save(
                    checkpoint,
                    os.path.join(
                        save_dir,
                        "checkpoints",
                        f"{trainer.config['wandb_name']}_{step}.pt",
                    ),
                )

        if backup_steps is not None and step % backup_steps == 0:
            if save_dir is not None:
                t.save(
                    {
                        "step": step,
                        "ae": trainer.ae.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "config": trainer.config,
                        "norm_factor": norm_factor,
                    },
                    os.path.join(save_dir, "ae.pt"),
                )

        with autocast_context:
            trainer.update(step, act)

    path = None
    if save_dir is not None:
        final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
        final_path = os.path.join(save_dir, f"{trainer.config['wandb_name']}.pt")
        print(final_path)

        t.save(final, final_path)
        path = final_path

    if use_wandb:
        log_queue.put("DONE")
        wandb_process.join()

    return path
