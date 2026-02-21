from dictionary_learning.evaluation import evaluate

from soft_sae.soft_top_k import SoftTopKSAE
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

seed = 42
steps = 40_000

trainer_config = {
    "trainer": MatryoshkaBatchTopKTrainer,
    "activation_dim": 512,
    "dict_size": 4096,
    "device": "cuda",
    "steps": steps,
    "layer": -1,
    "lm_name": "CLIP",
    "k": 256,
    "seed": seed,
    "auxk_alpha": 0.044,
    "dead_feature_threshold": 760_000,
    "decay_start": 7_000,
    "k_anneal_steps": 3_700,
    "lr": 0.0007,
    "warmup_steps": 4_000,
    "group_fractions": [0.1, 0.4, 0.5]
}

with wandb.init("st0pien-default-team", project="SoftSAE", name="MSAE_baseline") as run:
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

    trained_sae = MatryoshkaBatchTopKSAE.from_pretrained(
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
