#!/usr/bin/env python3
"""
Convert a JSONL (with instruction,input,output,metadata) into a CSV for human review.

Usage:
  python export_for_review.py --input sft_dataset.generated.jsonl --output review.csv

The CSV will contain columns:
  index, source_filename, title, instruction, input, output, metadata_review_status (if present)

Use a spreadsheet or Label Studio to review and update a 'review_status' column (approved/needs_edit)
then re-import/merge edits back into the JSONL (simple manual process).
"""
import argparse
import csv
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_fine_tune.utils import load_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    items = load_jsonl(Path(args.input))
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "source_filename", "title", "instruction", "input", "output", "metadata_review_status"])
        for i, it in enumerate(items):
            meta = it.get("metadata", {})
            review_status = meta.get("review_status", "")
            writer.writerow([
                i,
                meta.get("source_filename", ""),
                meta.get("title", ""),
                it.get("instruction", "").replace("\n", " "),
                it.get("input", "").replace("\n", " "),
                it.get("output", "").replace("\n", " "),
                review_status
            ])
    print(f"Wrote review CSV to {args.output}. Open it in a spreadsheet and add 'approved'/'needs_edit' in metadata_review_status column.")

if __name__ == "__main__":
    main()