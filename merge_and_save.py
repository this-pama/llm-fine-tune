from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_dir = "./lora_adapter"
base_model_name = "gpt2"

# Load tokenizer (with PAD token, so vocab size = 50258)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
vocab_size = len(tokenizer)

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Resize base model's embeddings to match tokenizer
model.resize_token_embeddings(vocab_size)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_dir)

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print("Merged model saved to", adapter_dir)