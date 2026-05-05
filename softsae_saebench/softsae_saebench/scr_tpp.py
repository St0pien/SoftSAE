from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.scr_and_tpp.main import run_eval
from sae_list import selected_saes
import sae_bench.sae_bench_utils.activation_collection as activation_collection




config = ScrAndTppEvalConfig(
    model_name="google/gemma-2-2b",
    perform_scr=False
)

config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE["gemma-2-2b"]
config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE["gemma-2-2b"]

print("[+] Working now")
run_eval(config, selected_saes, device="cuda", output_path="scr_results", save_activations=True)