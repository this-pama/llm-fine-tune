#!/usr/bin/env python3
"""
Quick validator for the input files to check compatibility with data_prep.py.

Usage:
  python validate_inputs.py --input_dir data_inputs --max_examples 3

What it does:
- Scans files in input_dir
- For each file attempts to load it depending on extension (.json/.jsonl/.csv/.md/.txt/other)
- For structured formats (json/csv) finds candidate text fields using heuristics:
  ("content", "full_text", "title", "text", "description")
- Prints inferred source_type by filename and a short sample of extracted text (or error)
- Exits with code 0 (ok) or 2 (issues found)
"""
import argparse
import json
import csv
from pathlib import Path
import sys

TEXT_FIELD_CANDIDATES = ("content", "full_text", "title", "text", "description")

def infer_source_type(filename: str) -> str:
    lower = filename.lower()
    if "solution" in lower or "solution_platform" in lower or "solutions" in lower:
        return "solution_platform"
    if "experiment" in lower or "experiment_platform" in lower:
        return "experiment_platform"
    if "action" in lower or "action_plan" in lower:
        return "action_plan_platform"
    if "blog" in lower or "blogs" in lower or "publication" in lower:
        return "blogs"
    return "unknown"

def try_load_text_from_json(path: Path):
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"JSON parse error: {e}"}
    items = []
    if isinstance(raw, dict):
        # if top-level has a list value, try to extract that first
        for k, v in raw.items():
            if isinstance(v, list):
                raw = v
                break
        else:
            raw = [raw]
    if isinstance(raw, list):
        for obj in raw[:5]:
            if not isinstance(obj, dict):
                items.append({"sample": str(obj)[:400]})
                continue
            # heuristic find candidate fields
            found = None
            for field in TEXT_FIELD_CANDIDATES:
                if field in obj and obj[field]:
                    found = obj[field]
                    break
            if not found:
                # join significant string fields
                pieces = []
                for k, v in obj.items():
                    if isinstance(v, str) and len(v) > 30:
                        pieces.append(f"{k}: {v[:200]}")
                found = "\n\n".join(pieces) if pieces else ""
            items.append({"sample": (found or "")[:800]})
        return {"items": items, "count": len(raw)}
    else:
        return {"error": "Unexpected JSON shape"}

def try_load_text_from_csv(path: Path):
    rows = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for i, r in enumerate(reader):
                if i >= 5:
                    break
                # find candidate field
                found = None
                for field in TEXT_FIELD_CANDIDATES:
                    if field in r and r[field]:
                        found = r[field]
                        break
                if not found:
                    pieces = []
                    for k, v in r.items():
                        if isinstance(v, str) and len(v) > 30:
                            pieces.append(f"{k}: {v[:200]}")
                    found = "\n\n".join(pieces) if pieces else ""
                rows.append({"sample": (found or "")[:800]})
        return {"items": rows, "count": "unknown (csv rows read)"}
    except Exception as e:
        return {"error": f"CSV read error: {e}"}

def try_load_text_from_textfile(path: Path):
    try:
        txt = path.read_text(encoding="utf-8")
        return {"sample": txt[:2000], "length": len(txt)}
    except Exception as e:
        return {"error": f"Text read error: {e}"}

def analyze_file(path: Path, max_examples: int = 3):
    ext = path.suffix.lower()
    info = {"filename": path.name, "ext": ext, "source_type": infer_source_type(path.name)}
    if ext in (".json", ".jsonl"):
        res = try_load_text_from_json(path)
        info.update(res)
    elif ext in (".csv",):
        res = try_load_text_from_csv(path)
        info.update(res)
    elif ext in (".md", ".txt"):
        res = try_load_text_from_textfile(path)
        info.update(res)
    else:
        # attempt to read as text anyway
        try:
            txt = path.read_text(encoding="utf-8")
            info.update({"sample": txt[:2000], "length": len(txt)})
        except Exception as e:
            info.update({"error": f"Unsupported extension and read failed: {e}"})
    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data_inputs")
    parser.add_argument("--max_examples", type=int, default=3)
    args = parser.parse_args()

    p = Path(args.input_dir)
    if not p.exists() or not p.is_dir():
        print(f"Input directory {p} not found.", file=sys.stderr)
        sys.exit(2)

    files = sorted([x for x in p.iterdir() if x.is_file()])
    if not files:
        print(f"No files found in {p}", file=sys.stderr)
        sys.exit(2)

    issues = []
    print(f"Found {len(files)} files in {p}\n")
    for f in files:
        print("=== File:", f.name, "(", f.suffix, ") ===")
        info = analyze_file(f, max_examples=args.max_examples)
        if "error" in info:
            print("  ERROR:", info["error"])
            issues.append(f.name)
        else:
            print("  inferred_source_type:", info.get("source_type"))
            if "count" in info:
                print("  items/count:", info.get("count"))
            if "items" in info:
                for i, it in enumerate(info["items"][:args.max_examples]):
                    sample = it.get("sample", "").strip()
                    print(f"  sample[{i}]:", (sample[:400] + ("..." if len(sample) > 400 else "")) if sample else "(empty)")
            elif "sample" in info:
                s = info["sample"].strip()
                print("  sample:", (s[:1000] + ("..." if len(s) > 1000 else "")) if s else "(empty)")
        print()
    if issues:
        print("Some files had issues. Examples:", issues, file=sys.stderr)
        print("If you see errors about nested JSON, atypical field names, or binary files (PDF), you will need to adjust data_prep.py or pre-convert files.")
        sys.exit(2)
    print("Basic validation passed — data_prep.py's default heuristics should be able to read these files.")
    sys.exit(0)

if __name__ == "__main__":
    main()