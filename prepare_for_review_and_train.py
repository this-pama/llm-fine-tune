#!/usr/bin/env python3
"""
Prepare CSV for review and a filtered SFT JSONL for training.

Usage:
  python prepare_for_review_and_train.py --input sft_dataset.generated.jsonl --review review.csv --train train_sft.jsonl

Options:
  --emb-check    try to compute embedding similarity (requires sentence-transformers)
  --emb-threshold FLOAT   threshold (default 0.18) for similarity to pass
  --required-headings "Summary:,Key lessons:,Recommended actions:"  comma-separated
  --sample N   include only first N examples in the review CSV (for quick checks)
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv
import sys
import re

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def structural_check(text: str, required_headings: List[str]) -> List[str]:
    missing = []
    for h in required_headings:
        if re.search(re.escape(h), text or "", flags=re.IGNORECASE) is None:
            missing.append(h)
    return missing

def try_load_emb_model(name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer, util as st_util  # type: ignore
        m = SentenceTransformer(name)
        return m, st_util
    except Exception as e:
        print(f"Could not load sentence-transformers ({e}). Embedding checks disabled.", file=sys.stderr)
        return None, None

def compute_similarity(model, util, a: str, b: str) -> Optional[float]:
    try:
        ea = model.encode(a, convert_to_tensor=True)
        eb = model.encode(b, convert_to_tensor=True)
        sim = util.cos_sim(ea, eb).item()
        return float(sim)
    except Exception as e:
        print(f"Embedding similarity failed: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--review", required=True, help="Output CSV for review")
    parser.add_argument("--train", required=True, help="Output filtered JSONL for training")
    parser.add_argument("--emb-check", action="store_true")
    parser.add_argument("--emb-threshold", type=float, default=0.18)
    parser.add_argument("--required-headings", type=str, default="Summary:,Key lessons:,Recommended actions:")
    parser.add_argument("--sample", type=int, default=0, help="If >0, limit review CSV to first N examples")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print("Input not found:", input_path, file=sys.stderr)
        raise SystemExit(1)

    data = load_jsonl(input_path)
    req_headings = [h.strip() for h in args.required_headings.split(",") if h.strip()]

    emb_model = None
    emb_util = None
    if args.emb_check:
        emb_model, emb_util = try_load_emb_model()
        if emb_model is None:
            print("Continuing without embeddings.", file=sys.stderr)

    # Prepare review CSV
    with open(args.review, "w", newline="", encoding="utf-8") as csvfh:
        writer = csv.writer(csvfh)
        writer.writerow(["idx", "instruction", "input", "output_preview", "missing_headings", "embedding_similarity", "generated_by", "generated_at", "note_review_status"])
        count = 0
        for i, ex in enumerate(data):
            if args.sample and count >= args.sample:
                break
            out = ex.get("output", "") or ""
            missing = structural_check(out, req_headings)
            sim = None
            if emb_model:
                src = (ex.get("input") or "")[:2000]
                sim = compute_similarity(emb_model, emb_util, src, out) if out else None
            writer.writerow([
                i,
                (ex.get("instruction") or "")[:2000],
                (ex.get("input") or "")[:2000],
                (out or "")[:500].replace("\n", "\\n"),
                ";".join(missing),
                "" if sim is None else f"{sim:.4f}",
                ex.get("metadata", {}).get("generated_by", ""),
                ex.get("metadata", {}).get("generated_at", ""),
                ""
            ])
            count += 1

    # Prepare training JSONL
    train_out = []
    for i, ex in enumerate(data):
        out = ex.get("output", "") or ""
        if not out:
            continue
        missing = structural_check(out, req_headings)
        if missing:
            # skip examples failing required headings
            continue
        if emb_model:
            src = (ex.get("input") or "")[:2000]
            sim = compute_similarity(emb_model, emb_util, src, out) if out else None
            if sim is not None and sim < args.emb_threshold:
                continue
        # Build SFT record: keep instruction, input, output
        sft = {
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "output": out,
            "metadata": ex.get("metadata", {})
        }
        train_out.append(sft)

    # Write train JSONL
    with open(args.train, "w", encoding="utf-8") as fh:
        for rec in train_out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote review CSV: {args.review}")
    print(f"Wrote training JSONL ({len(train_out)} examples): {args.train}")

if __name__ == "__main__":
    main()