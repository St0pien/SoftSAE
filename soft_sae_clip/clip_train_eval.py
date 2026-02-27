from custom_core.evaluation import evaluate

from soft_top_k import SoftTopKSAE, SoftTopKTrainer
import torch
from custom_core.npy_buffer import NpyActivationBuffer
from custom_core.training import ActivationsNormalization, trainSAE
from baselines.batch_topk import BatchTopKTrainer, BatchTopKSAE
from baselines.topk import TopKTrainer, AutoEncoderTopK
from baselines.matryoshka_batch_topk import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
import wandb
import argparse
import json

architectures = {
    "TopK": {"trainer": TopKTrainer, "model": AutoEncoderTopK},
    "BatchTopK": {"trainer": BatchTopKTrainer, "model": BatchTopKSAE},
    "MSAE": {"trainer": MatryoshkaBatchTopKTrainer, "model": MatryoshkaBatchTopKSAE},
    "SoftSAE": {"trainer": SoftTopKTrainer, "model": SoftTopKSAE},
}


def main():
    parser = argparse.ArgumentParser("clip_baseline")
    parser.add_argument("-a", "--architecture", choices=architectures.keys())
    parser.add_argument(
        "-dt", "--data-train", help="Path to train data npy", required=True
    )
    parser.add_argument(
        "-dv", "--data-val", help="Path to validation data npy", required=True
    )
    parser.add_argument(
        "-c", "--config", help="Path to json config file", required=True
    )
    parser.add_argument("-s", "--save-dir")
    parser.add_argument("--seed", default=42)
    parser.add_argument("-n", help="Number of steps to train", default=40_000, type=int)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    with open(args.config) as config_file:
        trainer_config = json.load(config_file)

    trainer_config["trainer"] = architectures[args.architecture]["trainer"]
    trainer_config["seed"] = args.seed
    trainer_config["steps"] = args.n
    print(trainer_config)

    wb_name = f"{args.architecture}" if args.wandb_name is None else args.wandb_name
    with wandb.init("st0pien-default-team", project="SoftSAE", name=wb_name) as run:
        buffered_data = NpyActivationBuffer(
            args.data_train,
            npy_length=2_905_954,
            d_submodule=512,
            ctx_len=1,
            out_batch_size=8192,
            device="cuda",
            dtype=torch.float32,
            seed=args.seed,
        )

        path = trainSAE(
            data=buffered_data,
            steps=args.n,
            trainer_config=trainer_config,
            use_wandb=True,
            log_steps=50,
            save_dir="results/checkpoints",
            normalize_activations=ActivationsNormalization.SCALE_SHIFT,
            run=run,
        )

        buffered_val_data = NpyActivationBuffer(
            args.data_val,
            npy_length=1_281_167,
            d_submodule=512,
            ctx_len=1,
            out_batch_size=8192,
            device="cuda",
            dtype=torch.float32,
            seed=args.seed,
        )

        trained_sae = architectures[args.architecture]["model"].from_pretrained(
            path,
            device="cuda",
        )

        evals = evaluate(
            trained_sae,
            activations=buffered_val_data,
            device="cuda",
            n_batches=1000,
            batch_size=4096,
            normalize_batch=True,
        )

        print(evals)
        run.summary.update({f"test/{key}": value for key, value in evals.items()})


if __name__ == "__main__":
    main()
