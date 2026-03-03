from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning.utils import truncate_model, hf_mixed_dataset_to_generator


model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
)
model = model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
)

text = "Describe the books of William Shaeakspear to me"
tokens = tokenizer(text, return_tensors="pt")

print(tokens)
resp_tokens = model.generate(**{k: tokens[k].cuda() for k in tokens})
print(resp_tokens)
response = tokenizer.decode(resp_tokens[0])
print(response)
