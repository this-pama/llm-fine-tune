#!/usr/bin/env python3
"""
LoRA + PEFT fine-tuning script for a large causal LM (e.g., Llama-3-8b).

This script expects a JSONL dataset with fields: instruction, input, output, and uses
the following prompt format for causal LM instruction-tuning:

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}

It creates labels where tokens corresponding to the prompt/instruction/input are set to -100
so only the response tokens contribute to the loss.

Notes / prerequisites:
- Run on a GPU instance (A100/H100 recommended for Llama-3-8b).
- Install: transformers, datasets, accelerate, peft, bitsandbytes, sentencepiece, safetensors
- Ensure you have access to the exact Llama-3.8b HF repo and have accepted license/terms.
- Replace MODEL_NAME with the Hugging Face repo name you will use.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          default_data_collator)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, get_peft_model_state_dict

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)

def build_prompt(example):
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    resp = example.get("output", "")
    return PROMPT_TEMPLATE.format(instruction=instruction, input=inp, response=resp)

def tokenize_and_mask(tokenizer, examples, max_length=2048, response_max_length=512):
    input_ids_list = []
    labels_list = []
    for ex in examples:
        full = build_prompt(ex)
        enc = tokenizer(full, truncation=True, max_length=max_length, padding=False)
        input_ids = enc["input_ids"]
        prompt_only = PROMPT_TEMPLATE.format(instruction=ex.get("instruction", ""),
                                             input=ex.get("input", ""),
                                             response="")
        prompt_enc = tokenizer(prompt_only, truncation=True, max_length=max_length, padding=False)
        prompt_len = len(prompt_enc["input_ids"])
        labels = [-100] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]
        if sum(1 for x in labels if x != -100) > response_max_length:
            resp_indices = [i for i, v in enumerate(labels) if v != -100]
            keep = resp_indices[-response_max_length:]
            new_labels = [-100] * len(labels)
            for j in keep:
                new_labels[j] = labels[j]
            labels = new_labels
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    batch = tokenizer.pad({"input_ids": input_ids_list, "labels": labels_list}, return_tensors="pt")
    return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sft_dataset.curated.jsonl", help="JSONL file (curated, no empty outputs)")
    parser.add_argument("--model_name", type=str, required=True, help="HF model repo (e.g., meta-llama/Llama-3-8b)")
    parser.add_argument("--output_dir", type=str, default="lora-llama3-8b", help="Where to save LoRA adapters")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=1, help="same as per_device if not using gradient accumulation")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--response_max_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.dataset, split="train")
    print("Loaded dataset:", ds)

    empty_outputs = sum(1 for x in ds if not x.get("output"))
    if empty_outputs > 0:
        raise SystemExit(f"Found {empty_outputs} examples with empty 'output'. Provide targets before training.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    print("Loading model (this may take a while). Ensure you have proper HF access and license accepted.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        offload_folder="offload",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print("LoRA modules added. Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    records = ds[:]
    batches = []
    batch_size = args.micro_batch_size
    for i in range(0, len(records), batch_size):
        chunk = records[i:i+batch_size]
        batches.append(tokenize_and_mask(tokenizer, chunk, max_length=args.max_length, response_max_length=args.response_max_length))

    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, batches):
            self.examples = []
            for b in batches:
                for i in range(b["input_ids"].size(0)):
                    self.examples.append({
                        "input_ids": b["input_ids"][i],
                        "attention_mask": b["attention_mask"][i],
                        "labels": b["labels"][i]
                    })
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return self.examples[idx]

    train_dataset = WrappedDataset(batches)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        push_to_hub=args.push_to_hub,
        remove_unused_columns=False,
        report_to="none"
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    print("Saving LoRA adapters to", args.output_dir)
    model.save_pretrained(args.output_dir)

    if args.push_to_hub:
        model.push_to_hub(args.output_dir)

if __name__ == "__main__":
    main()