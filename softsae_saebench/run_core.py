from dictionary_learning.trainers.soft_sae import SoftSAE
from wrappers import SAEBenchSoftSAE
from sae_bench.evals.core.main import multiple_evals


sae = SoftSAE.from_pretrained("../soft_sae_20_gemmma/ae.pt")


selected_saes = [("softsae_20_gemma", SAEBenchSoftSAE(sae))]

multiple_evals(selected_saes, 10, 1)
