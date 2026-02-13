"""
Training dictionaries
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

from .dictionary import AutoEncoder
from .evaluation import evaluate
from .trainers.standard import StandardTrainer


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
    trainer_configs: list[dict],
    steps: int,
    use_wandb: bool = False,
    wandb_entity: str = "",
    wandb_project: str = "",
    save_steps: Optional[list[int]] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    activations_split_by_head: bool = False,
    transcoder: bool = False,
    run_cfg: dict = {},
    normalize_activations: ActivationsNormalization = ActivationsNormalization.NONE,
    verbose: bool = False,
    device: str = "cuda",
    autocast_dtype: t.dtype = t.float32,
    backup_steps: Optional[int] = None,
    stop_wandb_logging: bool = True,
) -> list[str] | Tuple[list[str], list[wandb.Run]]:
    """
    Train SAEs using the given trainers

    If normalize_activations is True, the activations will be normalized to have unit mean squared norm.
    The autoencoders weights will be scaled before saving, so the activations don't need to be scaled during inference.
    This is very helpful for hyperparameter transfer between different layers and models.

    Setting autocast_dtype to t.bfloat16 provides a significant speedup with minimal change in performance.
    """

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = (
        nullcontext()
        if device_type == "cpu"
        else t.autocast(device_type=device_type, dtype=autocast_dtype)
    )

    timestamp = int(datetime.timestamp(datetime.now()))

    trainers = []
    for i, config in enumerate(trainer_configs):
        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    wandb_processes = []
    wandb_runs = []
    log_queues = []

    if use_wandb:
        # Note: If encountering wandb and CUDA related errors, try setting start method to spawn in the if __name__ == "__main__" block
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
        # Everything should work fine with the default fork method but it may not be as robust
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            # Make sure wandb config doesn't contain any CUDA tensors
            wandb_config = {
                k: v.cpu().item() if isinstance(v, t.Tensor) else v
                for k, v in wandb_config.items()
            }
            wandb_run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                config=config,
                name=wandb_config["wandb_name"],
            )
            wandb_runs.append(wandb_run)
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(log_queue, wandb_run),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, trainer.config["wandb_name"]) for trainer in trainers
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            try:
                config["buffer"] = data.config
            except:
                pass
            with open(
                os.path.join(
                    dir, f"config_{trainer.config['wandb_name']}_{timestamp}.json"
                ),
                "w",
            ) as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    norm_factor, shift_factor = get_norm_shift_factor(
        data, steps=100, activation_dim=trainer.ae.activation_dim
    )

    if normalize_activations != ActivationsNormalization.SCALE_SHIFT:
        shift_factor = 0.0
    if normalize_activations == ActivationsNormalization.NONE:
        norm_factor = 1.0

    for trainer in trainers:
        trainer.config["norm_factor"] = norm_factor
        if normalize_activations == ActivationsNormalization.SCALE_SHIFT:
            trainer.config["shift_factor_mean"] = shift_factor.mean().item()
        trainer.ae.set_normalization(norm_factor, shift_factor)

    for step, act in enumerate(tqdm(data, total=steps)):
        act = act.to(dtype=autocast_dtype)

        if step >= steps:
            break

        # logging
        if (use_wandb or verbose) and step % log_steps == 0:
            log_stats(
                trainers,
                step,
                act,
                activations_split_by_head,
                transcoder,
                log_queues=log_queues,
                verbose=verbose,
            )

        # saving
        if save_steps is not None and step in save_steps:
            for dir, trainer in zip(save_dirs, trainers):
                if dir is None:
                    continue

                if not os.path.exists(os.path.join(dir, "checkpoints")):
                    os.mkdir(os.path.join(dir, "checkpoints"))

                checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                t.save(
                    checkpoint,
                    os.path.join(
                        dir,
                        "checkpoints",
                        f"{trainer.config['wandb_name']}_{step}_{timestamp}.pt",
                    ),
                )

        # backup
        if backup_steps is not None and step % backup_steps == 0:
            for save_dir, trainer in zip(save_dirs, trainers):
                if save_dir is None:
                    continue
                # save the current state of the trainer for resume if training is interrupted
                # this will be overwritten by the next checkpoint and at the end of training
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

        # training
        for trainer in trainers:
            with autocast_context:
                trainer.update(step, act)

    paths = []
    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if save_dir is not None:
            final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
            final_path = os.path.join(
                save_dir, f"{trainer.config['wandb_name']}_{timestamp}.pt"
            )
            print(final_path)

            t.save(final, final_path)
            paths.append(final_path)

    if use_wandb and stop_wandb_logging:
        for run in wandb_runs:
            run.finish()

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()

    if use_wandb and not stop_wandb_logging:
        return paths, wandb_runs

    return paths
