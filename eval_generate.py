#!/usr/bin/env python3
"""
Quick script to generate text using your model + LoRA adapter.

Usage:
  python eval_generate.py --adapter_dir ./lora_adapter --prompt "### Instruction:\nSummarize\n\n### Input:\nClimate change is..." --max_new_tokens 100
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.95)
    return p.parse_args()

def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(args.adapter_dir, device_map="auto")
    if device == "mps":
        try: model.to(torch.device("mps"))
        except Exception: pass
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n=== GENERATED ===\n")
    print(text)
    print("\n=================\n")

if __name__ == "__main__":
    main()