from dictionary_learning.buffer import ActivationBuffer
from transformers import AutoModelForCausalLM
import demo_config
import json
import dictionary_learning.utils as utils
from dictionary_learning.evaluation import evaluate

model_name = "google/gemma-2-2b"
n_inputs = 200

ae_path = "soft_sae_v2_run_google_gemma-2-2b_soft_sae/resid_post_layer_12/trainer_2"
config_path = f"{ae_path}/config.json"

context_length = demo_config.LLM_CONFIG[model_name].context_length
llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
loss_recovered_batch_size = max(llm_batch_size // 5, 1)
sae_batch_size = loss_recovered_batch_size * context_length
dtype = demo_config.LLM_CONFIG[model_name].dtype

with open(config_path) as f:
    config = json.load(f)

layer = config["trainer"]["layer"]

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=dtype)
model = utils.truncate_model(model, layer)

buffer_size = n_inputs
n_batches = n_inputs // loss_recovered_batch_size

generator = utils.hf_dataset_to_generator("monology/pile-uncopyrighted")

input_strings = []
for i, example in enumerate(generator):
    input_strings.append(example)
    if i > n_inputs * 5 * llm_batch_size:
        break

dictionary, config = utils.load_dictionary(ae_path, device="cuda")
dictionary = dictionary.to(dtype=model.dtype)
submodule = utils.get_submodule(model, layer)
activation_dim = config["trainer"]["activation_dim"]

activation_buffer = ActivationBuffer(
    iter(input_strings),
    model,
    submodule,
    n_ctxs=buffer_size,
    ctx_len=context_length,
    refresh_batch_size=llm_batch_size,
    out_batch_size=sae_batch_size,
    io="out",
    d_submodule=activation_dim,
    device="cuda",
    remove_bos=demo_config.remove_bos,
    max_activation_norm_multiple=demo_config.max_activation_norm_multiple
)

eval_results = evaluate(
    dictionary,
    activation_buffer,
    context_length,
    loss_recovered_batch_size,
    io="out",
    device="cuda",
    n_batches=n_batches
)

print(eval_results)