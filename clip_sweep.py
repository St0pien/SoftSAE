from custom_core.npy_buffer import NpyActivationBuffer
from soft_sae.soft_top_k import SoftTopKTrainer, SoftTopKSAE
from dictionary_learning.training import trainSAE, ActivationsNormalization
from dictionary_learning.evaluation import evaluate
import torch

buffered_data = NpyActivationBuffer(
    "../TopKSAE-research/data/cc3m_ViT-B~16_train_image_2905954_512.npy",
    npy_length=2_905_954,
    d_submodule=512,
    ctx_len=1,
    out_batch_size=8192,
    device="cuda",
    dtype=torch.float32,
)

steps = 20_000

trainer_config = {
    "trainer": SoftTopKTrainer,
    "activation_dim": 512,
    "dict_size": 4096,
    "device": "cuda",
    "steps": steps,
    "layer": -1,
    "lm_name": "CLIP",
    "warmup_steps": 1000,
    "decay_start":  4000,
    "k": 64,
}

paths, runs = trainSAE(
    data=buffered_data,
    steps=steps,
    trainer_configs=[trainer_config],
    use_wandb=True,
    log_steps=50,
    wandb_entity="st0pien-default-team",
    wandb_project="SoftSAE",
    save_dir="results/checkpoints",
    normalize_activations=ActivationsNormalization.SCALE_SHIFT,
    device="cuda",
    stop_wandb_logging=False,
)

buffered_val_data = NpyActivationBuffer(
    "../TopKSAE-research/data/imagenet_ViT-B~16_train_image_1281167_512.npy",
    npy_length=1_281_167,
    d_submodule=512,
    ctx_len=1,
    out_batch_size=8192,
    device="cuda",
    dtype=torch.float32,
)

trained_sae = SoftTopKSAE.from_pretrained(paths[0], device="cuda")

evals = evaluate(
    trained_sae,
    activations=buffered_val_data,
    device="cuda",
    n_batches=500,
)

runs[0].summary.update({f"test/{key}": value for key, value in evals.items()})
runs[0].finish()
