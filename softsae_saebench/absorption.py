from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.main import run_eval
from sae_list import selected_saes

config = AbsorptionEvalConfig(
    model_name="google/gemma-2-2b"
)


run_eval(config, selected_saes, device="cuda", output_path="absorption_results")