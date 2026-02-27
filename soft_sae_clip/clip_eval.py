from dictionary_learning.evaluation import evaluate

from soft_top_k import SoftTopKSAE
import torch
from custom_core.npy_buffer import NpyActivationBuffer
from custom_core.training import ActivationsNormalization, trainSAE
from baselines.batch_topk import BatchTopKTrainer, BatchTopKSAE
from baselines.topk import AutoEncoderTopK
from baselines.matryoshka_batch_topk import MatryoshkaBatchTopKSAE
from clip_train_eval import architectures
import wandb
import argparse


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        "clip_eval", description="Run evalution on CLIP embeddings"
    )
    parser.add_argument("-a", "--architecture", choices=architectures.keys())
    parser.add_argument("-p", "--path", help="Path to model weights", required=True)
    parser.add_argument(
        "-d", "--data", help="Path to validation data npy file", required=True
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    buffered_val_data = NpyActivationBuffer(
        args.data,
        npy_length=1_281_167,
        d_submodule=512,
        ctx_len=1,
        out_batch_size=8192,
        device="cuda",
        dtype=torch.float32,
        seed=args.seed,
    )

    trained_sae = architectures[args.architecture]["model"].from_pretrained(
        args.path, device=device
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


if __name__ == "__main__":
    main()
