#!/usr/bin/env python3
"""
Top-level wrapper script for generation. Adds --hf-fallback-model CLI flag to
override the HF_FALLBACK_MODEL env var for a single run.
"""
import argparse
import json
from pathlib import Path
from src.llm_fine_tune.generate import generate_with_retry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL dataset")
    parser.add_argument("--output", required=True, help="Output JSONL with generated outputs")
    parser.add_argument("--model", type=str, default="llama3.1", help="Primary model (e.g. ollama model name)")
    parser.add_argument("--use_hf_fallback", action="store_true", help="Allow HF fallback when primary provider fails")
    parser.add_argument("--hf-fallback-model", dest="hf_fallback_model", type=str, default=None,
                        help="Explicit HuggingFace fallback model name (overrides HF_FALLBACK_MODEL env var).")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    out_path = Path(args.output)
    data = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            data.append(json.loads(line))

    results = []
    for i, rec in enumerate(data):
        prompt = rec.get("prompt") or rec.get("instruction", "") + "\n\n" + (rec.get("input") or "")
        gen = generate_with_retry(prompt, model=args.model,
                                  use_hf_fallback=args.use_hf_fallback,
                                  hf_fallback_model=args.hf_fallback_model)
        rec["output"] = gen
        results.append(rec)

    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()