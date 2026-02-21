from dictionary_learning.evaluation import evaluate

from soft_sae.soft_top_k import SoftTopKSAE
import torch
from custom_core.npy_buffer import NpyActivationBuffer
from custom_core.training import ActivationsNormalization, trainSAE
from baselines.batch_topk import BatchTopKTrainer, BatchTopKSAE
import wandb

seed = 42
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

# trained_sae = BatchTopKSAE.from_pretrained(
#     "results/checkpoints/BatchTopKSAE/BatchTopKSAE.pt",
#     device="cuda",
# )

trained_sae = SoftTopKSAE.from_pretrained(
    "results/checkpoints/vague-sweep-90_CLIP_512_4096_256/vague-sweep-90_CLIP_512_4096_256.pt",
    device="cuda"
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
