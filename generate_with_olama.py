#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from src.llm_fine_tune.generate import generate_with_retry

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--model", default="llama3.1")
parser.add_argument("--use_hf_fallback", action="store_true")
parser.add_argument("--hf-fallback-model", dest="hf_fallback_model", default=None,
                    help="Explicit HF fallback model name (overrides HF_FALLBACK_MODEL env var)")
args = parser.parse_args()

input_path = Path(args.input)
if not input_path.exists():
    raise SystemExit("Input not found")

with open(input_path, encoding="utf-8") as fh:
    data = [json.loads(l) for l in fh]

out = []
for rec in data:
    prompt = rec.get("prompt") or rec.get("instruction", "") + "\n\n" + (rec.get("input") or "")
    gen = generate_with_retry(prompt, model=args.model, use_hf_fallback=args.use_hf_fallback,
                              hf_fallback_model=args.hf_fallback_model)
    rec["output"] = gen
    out.append(rec)

with open(args.output, "w", encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")