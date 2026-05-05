import json
import torch
from transformer_lens import HookedTransformer
from wrappers import SoftSAE, SAEBenchSoftSAE

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "gemma-2-2b"
HOOK_NAME = "blocks.12.hook_resid_post"

prompts = [
    "The capital of France is",
    "Machine learning is",
]

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# LOAD MODEL + SAE
# -----------------------
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

sae = SoftSAE.from_pretrained(
    "../soft_sae_160_320/resid_post_layer_12/trainer_1/ae.pt",
    device=device,
)

sae = sae.to(device)

sae = SAEBenchSoftSAE(sae)


# -----------------------
# RUN
# -----------------------
results = {}

with torch.no_grad():
    tokens = model.to_tokens(prompts)

    _, cache = model.run_with_cache(
        tokens, names_filter=[HOOK_NAME], return_type="logits"
    )

    resid = cache[HOOK_NAME]  # [batch, seq, d_model]

    # run SAE
    features = sae.encode(resid)  # [batch, seq, d_sae]

    # L0 per prompt (count active features across sequence)
    l0_per_prompt = (features != 0).sum(dim=(1, 2)).tolist()

    for p, l0 in zip(prompts, l0_per_prompt):
        results[p] = l0


# -----------------------
# SAVE
# -----------------------
with open("l0_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(results)
