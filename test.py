from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.training import trainSAE
from soft_sae.soft_top_k import SoftTopKSAE, SoftTopKTrainer
from custom_core.npy_buffer import NpyActivationBuffer
import torch


buffer = NpyActivationBuffer(
    "../TopKSAE-research/data/cc3m_ViT-B~16_train_image_2905954_512.npy",
    npy_length=2_905_954,
    d_submodule=512,
    ctx_len=1,
    out_batch_size=8192,
    device="cuda",
    dtype=torch.float32
)



# trainer = TopKTrainer(
#     10000,
#     activation_dim=512,
#     dict_size=4096,
#     k=64,
#     layer=-1,
#     lm_name="CLIP",
#     warmup_steps=2,
# )

# trainer_config = {
#     "trainer": TopKTrainer,
#     "dict_class": AutoEncoderTopK,
#     "activation_dim": 512,
#     "dict_size": 4096,
#     "lr": 1e-3,
#     "device": "cuda",
#     "steps": 10_000,
#     "layer": -1,
#     "lm_name": "CLIP",
#     "warmup_steps": 1000,
#     "k": 64,
# }

# trainSAE(
#     data=buffer,
#     trainer_configs=[trainer_config],
#     steps=2000,
#     use_wandb=True,
#     log_steps=10,
#     wandb_entity="st0pien-default-team",
#     wandb_project="SoftSAE",
# )


steps = 2_000

trainer_config = {
    "trainer": SoftTopKTrainer,
    "activation_dim": 512,
    "dict_size": 4096,
    "device": "cuda",
    "steps": steps,
    "layer": -1,
    "lm_name": "CLIP",
    "warmup_steps": 200,
    "k": 64,
}

trainSAE(
    data=buffer,
    steps=steps,
    trainer_configs=[trainer_config],
    use_wandb=True,
    log_steps=10,
    wandb_entity="st0pien-default-team",
    wandb_project="SoftSAE",
    save_dir="results/checkpoints",
)
