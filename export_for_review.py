#!/usr/bin/env python3
"""
export_for_review.py

Read a merged JSONL (one record per original CSV row, produced by merge_generated.py)
and write a CSV suitable for human review.

Example:
  python export_for_review.py --input final.jsonl --output review.csv

The script will:
- detect the merged/generated output field (merged_output, final_output, output, generated)
- flatten original_row or metadata (if present) into CSV columns
- write UTF-8 CSV with proper quoting
"""
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Set

COMMON_OUTPUT_KEYS = ("merged_output", "merged_text", "final_output", "output", "generated", "completion")

def detect_output_field(d: Dict[str, Any]) -> str:
    for k in COMMON_OUTPUT_KEYS:
        if k in d:
            return k
    # fallback: choose first non-metadata non-source key that looks like text
    for k, v in d.items():
        if k not in ("source_id", "original_row", "metadata") and isinstance(v, str) and len(v) > 0:
            return k
    return ""  # nothing found

def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    # source_id
    if "source_id" in rec:
        out["source_id"] = rec["source_id"]
    # merged/generated output
    out_field = detect_output_field(rec)
    if out_field:
        out["merged_output"] = rec.get(out_field, "")
    else:
        # leave blank if nothing found
        out["merged_output"] = ""

    # pick original metadata
    orig = rec.get("original_row", None) or rec.get("metadata", None)
    if isinstance(orig, dict):
        for k, v in orig.items():
            # convert nested structures to JSON strings
            if isinstance(v, (dict, list)):
                out[k] = json.dumps(v, ensure_ascii=False)
            else:
                out[k] = "" if v is None else v
    elif orig is not None:
        # string or other - put into original_row column
        out["original_row"] = json.dumps(orig, ensure_ascii=False) if not isinstance(orig, str) else orig

    # carry over any additional top-level scalar fields that look useful
    for k, v in rec.items():
        if k in ("source_id", "original_row", "metadata", out_field):
            continue
        if isinstance(v, (str, int, float, bool)) and k not in out:
            out[k] = v

    return out

def gather_headers(rows: List[Dict[str, Any]]) -> List[str]:
    headers: Set[str] = set()
    for r in rows:
        headers.update(r.keys())
    # prefer ordering: source_id, merged_output, then sorted remaining
    ordering = []
    if "source_id" in headers:
        ordering.append("source_id")
        headers.remove("source_id")
    if "merged_output" in headers:
        ordering.append("merged_output")
        headers.remove("merged_output")
    # keep original_row if present (stringified)
    if "original_row" in headers:
        ordering.append("original_row")
        headers.remove("original_row")
    ordering.extend(sorted(headers))
    return ordering

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Input merged JSONL (one record per original CSV row)")
    p.add_argument("--output", "-o", required=True, help="Output CSV file for review")
    p.add_argument("--output-columns", help="Comma-separated list of CSV columns to include (optional)")
    p.add_argument("--include-original", action="store_true", help="Include original_row flattened columns (default behavior)")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    records = []
    with in_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSONL line {i}: {e}")
                continue
            flat = flatten_record(rec)
            records.append(flat)

    if not records:
        raise SystemExit("No valid records found in input.")

    if args.output_columns:
        headers = [c.strip() for c in args.output_columns.split(",") if c.strip()]
    else:
        headers = gather_headers(records)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in records:
            # ensure all header keys exist in row
            row = {k: ("" if r.get(k) is None else r.get(k)) for k in headers}
            writer.writerow(row)

    print(f"Wrote {len(records)} rows to {out_path} with columns: {', '.join(headers)}")

if __name__ == "__main__":
    main()