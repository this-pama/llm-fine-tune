#!/usr/bin/env python3
"""
train_sft_mps_safe.py

Same as train_sft.py but:
- Avoids MPS cholesky error by performing embedding resize on CPU or enabling MPS fallback.
- Refuses to use bitsandbytes 8-bit on non-CUDA hosts.
- Keeps LoRA/PEFT flow.

Usage (quick smoke-test):
  PYTORCH_ENABLE_MPS_FALLBACK=1 python train_sft_mps_safe.py --train_file train_sft.jsonl --model_name_or_path gpt2 --output_dir ./lora_test

Or just run:
  python train_sft_mps_safe.py --train_file train_sft.jsonl --model_name_or_path gpt2 --output_dir ./lora_test
which will perform the safe CPU resize when needed.
"""
import os
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

# It's OK to set the env var here to enable CPU fallback for MPS when needed.
# You can remove or override this if you prefer explicit CLI env var.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from transformers import DataCollatorWithPadding

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

PROMPT_WRAPPER = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

@dataclass
class Example:
    instruction: str
    input: str
    output: str

class DataCollatorForSupervisedDataset:
    """
    Collate function for supervised fine-tuning with custom 'labels' padding.
    Pads input_ids, attention_mask, and labels to the longest in the batch.
    """
    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, batch):
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        labels = [x["labels"] for x in batch]
        batch_token = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=self.padding,
            return_tensors="pt"
        )
        max_length = batch_token["input_ids"].shape[1]
        # FIX: Convert labels tensor to list before padding
        labels_padded = [l.tolist() + [-100] * (max_length - len(l)) for l in labels]
        batch_token["labels"] = torch.tensor(labels_padded, dtype=torch.long)
        return batch_token

    
def load_jsonl_examples(path: Path) -> List[Example]:
    items: List[Example] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            items.append(Example(instruction=j.get("instruction",""), input=j.get("input",""), output=j.get("output","")))
    return items

class SFTDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer, cutoff_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.inputs = []
        self.prepare()

    def prepare(self):
        for ex in self.examples:
            prompt = PROMPT_WRAPPER.format(instruction=ex.instruction, input=ex.input)
            full = prompt + (ex.output or "")
            tokenized = self.tokenizer(full, truncation=True, max_length=self.cutoff_len, padding=False)
            input_ids = tokenized["input_ids"]
            prompt_tok = self.tokenizer(prompt, truncation=True, max_length=self.cutoff_len, padding=False)
            prompt_len = len(prompt_tok["input_ids"])
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            if len(labels) != len(input_ids):
                labels = labels[: len(input_ids)]
                if len(labels) < len(input_ids):
                    labels += [-100] * (len(input_ids) - len(labels))
            self.inputs.append({"input_ids": input_ids, "attention_mask": tokenized.get("attention_mask", [1]*len(input_ids)), "labels": labels})

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.inputs[idx].items()}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="JSONL file with instruction/input/output")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes + CUDA)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (or set HUGGINGFACE_HUB_TOKEN env var)")
    return parser.parse_args()

def safe_resize_token_embeddings_cpu(model, tokenizer):
    """
    If tokenizer and model vocab sizes differ, do the resize on CPU (avoids MPS-cholesky bug).
    """
    new_num_tokens = len(tokenizer)
    try:
        old_num = model.get_input_embeddings().weight.size(0)
    except Exception:
        old_num = None

    if old_num is not None and new_num_tokens == old_num:
        return

    # Move model to CPU, resize, then move back to original device.
    # For small models this is fine. For very large models this will be slow and may not be desirable.
    orig_device = next(model.parameters()).device
    try:
        model.to("cpu")
        model.resize_token_embeddings(new_num_tokens)
    finally:
        try:
            model.to(orig_device)
        except Exception:
            # moving back may fail on MPS for some ops — leave model on CPU if it fails
            print("Warning: moving model back to original device failed; model remains on CPU.")

def main():
    args = parse_args()
    train_path = Path(args.train_file)
    assert train_path.exists(), f"Train file not found: {train_path}"

    examples = load_jsonl_examples(train_path)
    if len(examples) == 0:
        raise SystemExit("No examples loaded from train file.")

    # Basic device selection (we won't assume CUDA on Mac)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print("Device selected:", device)
    if args.load_in_8bit and device != "cuda":
        print("Warning: --load_in_8bit requires CUDA/bitsandbytes — ignoring --load_in_8bit on this host.")
        args.load_in_8bit = False

    # Load tokenizer (support token/hf auth if provided)
    tokenizer_kwargs = {"use_fast": True}
    if args.hf_token:
        tokenizer_kwargs["use_auth_token"] = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)

    # If tokenizer lacks pad token, add it (this may require resizing embeddings)
    added_special = False
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        added_special = True

    # Load model (use device_map auto; low_cpu_mem_usage True for memory)
    model_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
    if args.hf_token:
        model_kwargs["use_auth_token"] = args.hf_token
    if args.fp16 and device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    if args.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, **model_kwargs)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    # If we added tokens, perform safe CPU resize (avoids MPS linalg op issue)
    if added_special:
        print("Resizing token embeddings on CPU to accommodate new special tokens...")
        safe_resize_token_embeddings_cpu(model, tokenizer)

    # Apply LoRA
    target_modules = ["q_proj", "v_proj"] if "llama" in args.model_name_or_path.lower() else None
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Dataset and training
    dataset = SFTDataset(examples, tokenizer, cutoff_len=args.cutoff_len)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16 and device == "cuda",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator)
    print("Starting training...")
    trainer.train()
    print("Saving adapter + tokenizer to", args.output_dir)
    # Merge LoRA weights into base model and save
    # If 'model' is still a PEFT model, merge and save
    if hasattr(model, "merge_and_unload"):
        base_model = model.merge_and_unload()
        base_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()