import sae_bench.sae_bench_utils.activation_collection as activation_collection
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.main import run_eval
from sae_list import selected_saes

config = RAVELEvalConfig(
    model_name="gemma-2-2b"
)

config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE["gemma-2-2b"]
config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE["gemma-2-2b"]

run_eval(config, selected_saes, device="cuda", output_path="ravel_results")