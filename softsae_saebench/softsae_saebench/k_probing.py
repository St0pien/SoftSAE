from sae_bench.evals.sparse_probing_sae_probes.eval_config import SparseProbingSaeProbesEvalConfig
from sae_bench.evals.sparse_probing_sae_probes.main import run_eval
from sae_list import selected_saes

config = SparseProbingSaeProbesEvalConfig(
    model_name="google/gemma-2-2b"
)


run_eval(config, selected_saes, device="cuda", output_path="sparse_probing_results")