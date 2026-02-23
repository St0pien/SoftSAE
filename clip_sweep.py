from dictionary_learning.evaluation import evaluate
import torch
import wandb
import argparse
from custom_core.npy_buffer import NpyActivationBuffer
from custom_core.training import trainSAE, ActivationsNormalization
from soft_sae.soft_top_k import SoftTopKTrainer, SoftTopKSAE

sweep_parameters = {
    # "k_loss_type": {"values": ["budget", "kl0"]},
    "k_loss_weight": {"min": 0.0, "max": 5.0},
    "lr": {"min": 0.0001, "max": 0.001},
    "auxk_alpha": {"min": 0.01, "max": 0.1},
    "dead_feature_threshold": {
        "min": 100_000,
        "max": 1_000_000,
        "distribution": "int_uniform",
    },
    "warmup_steps": {"min": 0, "max": 5_000, "distribution": "int_uniform"},
    "decay_start": {"min": 5_001, "max": 10_000, "distribution": "int_uniform"},
    "k_anneal_steps": {"min": 0, "max": 10_000, "distribution": "int_uniform"},
    "alpha_anneal_steps": {"min": 0, "max": 10_000, "distribution": "int_uniform"},
    "hard_topk_steps": {"min": 0, "max": 10_000, "distribution": "int_uniform"},
}

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "test/frac_variance_explained"},
    "parameters": sweep_parameters,
}

entity = "st0pien-default-team"
project = "SoftSAE"

steps = 40_000
seed = 42
shared_config = {
    "k_loss_type": "budget",
    "trainer": SoftTopKTrainer,
    "activation_dim": 512,
    "dict_size": 4096,
    "soft_topk_alpha": 0.0001,
    "device": "cuda",
    "steps": steps,
    "layer": -1,
    "lm_name": "CLIP",
    "k": 256,
    "seed": seed,
}


def sweep_run():
    with wandb.init() as run:
        run_config = dict(run.config)

        buffered_data = NpyActivationBuffer(
            "../TopKSAE-research/data/cc3m_ViT-B~16_train_image_2905954_512.npy",
            npy_length=2_905_954,
            d_submodule=512,
            ctx_len=1,
            out_batch_size=8192,
            device="cuda",
            dtype=torch.float32,
            seed=seed,
        )
        trainer_config = run_config | shared_config
        trainer_config["wandb_name"] = run.name

        path = trainSAE(
            data=buffered_data,
            steps=steps,
            trainer_config=trainer_config,
            use_wandb=True,
            log_steps=50,
            save_dir="results/checkpoints",
            normalize_activations=ActivationsNormalization.SCALE_SHIFT,
            run=run,
        )

        buffered_val_data = NpyActivationBuffer(
            "../TopKSAE-research/data/imagenet_ViT-B~16_train_image_1281167_512.npy",
            npy_length=1_281_167,
            d_submodule=512,
            ctx_len=1,
            out_batch_size=8192,
            device="cuda",
            dtype=torch.float32,
            seed=seed,
        )

        trained_sae = SoftTopKSAE.from_pretrained(path, device="cuda")

        evals = evaluate(
            trained_sae,
            activations=buffered_val_data,
            device="cuda",
            n_batches=100,
            batch_size=4096,
            normalize_batch=True,
        )

        run.summary.update({f"test/{key}": value for key, value in evals.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "clip_sweep.py", description="Run CLIP SoftSAE sweep"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "init", description="Initialize sweep in wandb and get sweep ID"
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--sweep-id", required=True)
    run_parser.add_argument("--agent-count", default=1, type=int)

    args = parser.parse_args()

    if args.command == "init":
        sweep_id = wandb.sweep(
            sweep=sweep_configuration, project=project, entity=entity
        )
        print(f"Wandb sweep initialized with ID:\n\t{sweep_id}")
    else:
        wandb.agent(
            args.sweep_id,
            project=project,
            entity=entity,
            function=sweep_run,
            count=args.agent_count,
        )
