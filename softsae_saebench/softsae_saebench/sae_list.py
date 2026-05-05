from wrappers import SAEBenchSAE
from soft_sae_patched import SoftSAEPatched
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE

sae_list = [
    # ("batchtopk_sweep_20_40_80_google_gemma-2-2b_batch_top_k/resid_post_layer_12/trainer_0/ae.pt", BatchTopKSAE),
    # ("batchtopk_sweep_20_40_80_google_gemma-2-2b_batch_top_k/resid_post_layer_12/trainer_1/ae.pt", BatchTopKSAE),
    # ("batchtopk_sweep_20_40_80_google_gemma-2-2b_batch_top_k/resid_post_layer_12/trainer_2/ae.pt", BatchTopKSAE),
    # ("batchtopk_sweep_160_320_google_gemma-2-2b_batch_top_k/resid_post_layer_12/trainer_0/ae.pt", BatchTopKSAE),
    # ("batchtopk_sweep_160_320_google_gemma-2-2b_batch_top_k/resid_post_layer_12/trainer_1/ae.pt", BatchTopKSAE),

    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_0/ae.pt", AutoEncoderTopK),
    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_1/ae.pt", AutoEncoderTopK),
    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_2/ae.pt", AutoEncoderTopK),
    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_3/ae.pt", AutoEncoderTopK),
    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_4/ae.pt", AutoEncoderTopK),

    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_5/ae.pt", MatryoshkaBatchTopKSAE),
    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_6/ae.pt", MatryoshkaBatchTopKSAE),
    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_7/ae.pt", MatryoshkaBatchTopKSAE),
    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_8/ae.pt", MatryoshkaBatchTopKSAE),
    # ("matryoshka_topk_sweep_google_gemma-2-2b_matryoshka_batch_top_k_top_k/resid_post_layer_12/trainer_9/ae.pt", MatryoshkaBatchTopKSAE),

    ("soft_sae_20_gemma/trainer_0/ae.pt", SoftSAEPatched),
    ("soft_sae_sweep_google_gemma-2-2b_soft_sae/resid_post_layer_12/trainer_0/ae.pt", SoftSAEPatched),
    ("soft_sae_sweep_80_google_gemma-2-2b_soft_sae/resid_post_layer_12/trainer_0/ae.pt", SoftSAEPatched),
    ("soft_sae_sweep_160_320_google_gemma-2-2b_soft_sae/resid_post_layer_12/trainer_0/ae.pt", SoftSAEPatched),
    ("soft_sae_sweep_160_320_google_gemma-2-2b_soft_sae/resid_post_layer_12/trainer_1/ae.pt", SoftSAEPatched),
]

selected_saes = []
for path, cls in sae_list:
    sae = cls.from_pretrained(path).cuda()
    selected_saes.append((f"{cls.__name__}_{sae.k}", SAEBenchSAE(sae)))
