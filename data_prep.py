#!/usr/bin/env python3
"""
Prepare instruction-style JSONL from your four DB exports, applying
different instruction templates depending on source type.

Updated to better handle the CSV structures you provided:
- Prefer 'full_text' column when present.
- Parse 'sections' column when it's a JSON string and extract nested text (items[].txt etc.).
- Strip HTML (uses bs4 if available, otherwise a regex fallback).

Usage:
  python data_prep.py --input_dir data_inputs --output sft_dataset.jsonl --max_chars 2000
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Union
import csv
import sys

# Immediately after imports (once) set the field size limit to avoid "field larger than field limit"
try:
    csv.field_size_limit(sys.maxsize)
except Exception:
    # some platforms may not accept sys.maxsize; try a very large int
    csv.field_size_limit(10 * 1024 * 1024)

# ---- Optional HTML cleaner ----
try:
    from bs4 import BeautifulSoup

    def strip_html(s: str) -> str:
        return BeautifulSoup(s, "html.parser").get_text(separator="\n")
except Exception:
    _TAG_RE = re.compile(r"<[^>]+>")

    def strip_html(s: str) -> str:
        return _TAG_RE.sub("", s)

# ---- Utilities ----
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n\s+\n+", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start + max_chars // 4:
                end = nl
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(end - overlap, end)
    return chunks

# ---- Loaders ----
def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def load_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # return list if top-level contains a list field
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    return data

def load_csv_simple(path: Path) -> List[dict]:
    """
    Read CSV robustly. Increase csv.field_size_limit and fall back to pandas if Python csv fails.
    """
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for r in reader:
                rows.append(r)
        return rows
    except csv.Error as e:
        # If csv parsing fails due to large fields, try a safer pandas read (engine='python')
        print(f"CSV csv.Error reading {path}: {e} — retrying with pandas (engine='python').")
        try:
            import pandas as pd
            # use engine='python' which is more tolerant of large fields
            df = pd.read_csv(path, engine="python", encoding="utf-8")
            # convert NaN to empty string and dict-ify
            df = df.fillna("")
            return df.to_dict(orient="records")
        except Exception as e2:
            print(f"Pandas fallback failed for {path}: {e2}")
            raise

# ---- Robust extraction for your CSV shapes ----
def _extract_from_sections_field(val: Union[str, list, dict]) -> List[str]:
    """
    When 'sections' column contains JSON (often a string), parse and recursively extract
    text-like fields (txt, title, html, body, content). Return list of text bits.
    """
    bits = []

    def rec(node):
        if node is None:
            return
        if isinstance(node, str):
            # try to detect JSON encoded string
            s = node.strip()
            # ignore short tokens
            if (s.startswith("{") or s.startswith("[")) and len(s) > 50:
                try:
                    parsed = json.loads(s)
                    rec(parsed)
                    return
                except Exception:
                    # not JSON, maybe HTML/text
                    cleaned = strip_html(s)
                    if cleaned.strip():
                        bits.append(cleaned)
                    return
            # plain string
            cleaned = strip_html(s)
            if cleaned.strip():
                bits.append(cleaned)
            return
        if isinstance(node, dict):
            # common keys that hold text
            for k in ("txt", "text", "title", "html", "body", "content", "lead", "instruction"):
                if k in node and node[k]:
                    rec(node[k])
            # some nodes have 'items' which is a list/array of inner nodes
            if "items" in node and node["items"]:
                rec(node["items"])
            # also some nodes contain nested arrays under keys like 'structure'
            if "structure" in node and node["structure"]:
                rec(node["structure"])
            # fallback: iterate values
            for v in node.values():
                if isinstance(v, (dict, list, str)):
                    rec(v)
            return
        if isinstance(node, list):
            for it in node:
                rec(it)
            return

    rec(val)
    return bits

def extract_text_from_record(record: Dict[str, Any]) -> str:
    """
    Heuristic extraction:
    1) Prefer specific fields: full_text, sections (JSON), content, body, text, description, note, summary, blog
    2) If sections is JSON-like string, parse and extract nested txt/title/html fields
    3) Fallback: join all long string fields
    """
    if not isinstance(record, dict):
        return str(record) if record else ""

    # 1) full_text preferred
    for field in ("full_text", "fulltext", "content", "body", "text", "description", "note", "summary", "blog"):
        if field in record and record[field]:
            val = record[field]
            if isinstance(val, str):
                cleaned = strip_html(val)
                if len(cleaned.strip()) > 20:
                    return cleaned

    # 2) handle 'sections' if present (JSON string or structured)
    if "sections" in record and record["sections"]:
        sec_val = record["sections"]
        try:
            # if it's a string that looks like JSON, parse it
            if isinstance(sec_val, str):
                sec_val_stripped = sec_val.strip()
                if sec_val_stripped.startswith("[") or sec_val_stripped.startswith("{"):
                    parsed = json.loads(sec_val_stripped)
                    bits = _extract_from_sections_field(parsed)
                    if bits:
                        return "\n\n".join(b.strip() for b in bits if b.strip())
            else:
                bits = _extract_from_sections_field(sec_val)
                if bits:
                    return "\n\n".join(b.strip() for b in bits if b.strip())
        except Exception:
            # fallback to treating sections as plain text
            cleaned = strip_html(str(sec_val))
            if len(cleaned.strip()) > 20:
                return cleaned

    # 3) fallback: check other candidate fields again but accept any long strings
    pieces = []
    for k, v in record.items():
        if isinstance(v, str) and len(v.strip()) > 40:
            # skip repeating CSV metadata columns like id if too short
            pieces.append(f"{k}: {strip_html(v).strip()}")
    if pieces:
        return "\n\n".join(pieces)

    # 4) ultimate fallback: join any string fields longer than 10 chars
    pieces = []
    for k, v in record.items():
        if isinstance(v, str) and len(v.strip()) > 10:
            pieces.append(f"{k}: {strip_html(v).strip()}")
    return "\n\n".join(pieces)

# ---- Source type inference + templates ----
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

INSTRUCTION_TEMPLATES = {
    "solution_platform": (
        "You are a knowledgeable assistant about grassroots innovations and local solutions."
        " Read the document and produce a structured summary with sections: "
        "'Problem / Context:', 'Solution description:', 'Who is involved / stakeholders:', "
        "'Key implementation details:', 'Scalability / replication potential:', 'Risks / constraints:', "
        "and 'Recommended next steps:'."
    ),
    "experiment_platform": (
        "You are an expert in field experiments for sustainable development."
        " From the document, extract: 'Experiment objective / hypothesis:', 'Design / methods:', "
        "'Key results:', 'What worked:', 'What didn't work:', 'Quantitative / qualitative evidence:', "
        "and 'Recommendations for future experiments or scale-up:'."
    ),
    "action_plan_platform": (
        "You are a practical advisor on implementing action plans."
        " Summarize: 'Action goal:', 'Learning questions:', 'Planned steps / timeline:', 'Resources needed / partners:', "
        "'Outcomes achieved (if noted):', 'Lessons learned:', and 'Follow-up actions / monitoring suggestions:'."
    ),
    "blogs": (
        "You are a reflective analyst synthesizing blog and publication insights."
        " Produce a concise 'Summary of reflection:', 'Key lessons / insights:', 'Policy or practice implications:', "
        "and 'Suggested further reading or actions:'."
    ),
    "unknown": (
        "You are a helpful assistant. Summarize the document focusing on key points and recommended actions."
    )
}

# ---- Creation of SFT examples ----
def create_examples_from_doc(source_filename: str, title: str, text: str,
                             max_chars: int = 2000) -> List[Dict]:
    source_type = infer_source_type(source_filename)
    template = INSTRUCTION_TEMPLATES.get(source_type, INSTRUCTION_TEMPLATES["unknown"])
    chunks = chunk_text(text, max_chars=max_chars)
    examples = []
    for i, c in enumerate(chunks):
        instruction = template
        input_text = f"Source file: {source_filename}\nTitle: {title or ''}\n\nDocument:\n{c}"
        examples.append({
            "instruction": instruction,
            "input": input_text,
            "output": "",  # leave empty for manual annotation or synthetic generation
            "metadata": {
                "source_type": source_type,
                "source_filename": source_filename,
                "title": title or "",
                "chunk_index": i,
                "chunk_chars": len(c)
            }
        })
    return examples

# ---- Main collection ----
def collect_docs_from_dir(input_dir: Path) -> List[Dict]:
    docs = []
    for p in sorted(input_dir.glob("*")):
        if p.is_dir():
            continue
        try:
            if p.suffix.lower() in (".json", ".jsonl"):
                items = load_json(p)
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            text = extract_text_from_record(it)
                            title = it.get("title") or it.get("name") or ""
                            if text:
                                docs.append({"file": p.name, "title": title, "text": clean_text(text)})
                else:
                    it = items
                    text = extract_text_from_record(it)
                    title = it.get("title") or ""
                    if text:
                        docs.append({"file": p.name, "title": title, "text": clean_text(text)})
            elif p.suffix.lower() in (".csv",):
                rows = load_csv_simple(p)
                for r in rows:
                    text = extract_text_from_record(r)
                    title = r.get("title") or r.get("name") or ""
                    if text:
                        docs.append({"file": p.name, "title": title, "text": clean_text(text)})
            elif p.suffix.lower() in (".md", ".txt"):
                text = load_text_file(p)
                if text and len(text) > 20:
                    docs.append({"file": p.name, "title": p.stem, "text": clean_text(text)})
            else:
                text = load_text_file(p)
                if text and len(text) > 20:
                    docs.append({"file": p.name, "title": p.stem, "text": clean_text(text)})
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    return docs

def write_jsonl(items: List[Dict], outfile: Path):
    with outfile.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data_inputs")
    parser.add_argument("--output", type=str, default="sft_dataset.jsonl")
    parser.add_argument("--max_chars", type=int, default=2000)
    parser.add_argument("--generate_targets", action="store_true",
                        help="Optionally generate synthetic outputs using OpenAI (requires OPENAI_API_KEY env var).")
    parser.add_argument("--max_gen", type=int, default=200,
                        help="Max number of examples to auto-generate targets for when --generate_targets is used.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory {input_dir} not found.")

    docs = collect_docs_from_dir(input_dir)
    print(f"Found {len(docs)} documents.")
    examples = []
    for d in docs:
        exs = create_examples_from_doc(d["file"], d.get("title", ""), d["text"], max_chars=args.max_chars)
        examples.extend(exs)
    print(f"Created {len(examples)} examples (outputs empty by default).")

    write_jsonl(examples, Path(args.output))
    print(f"Wrote {len(examples)} examples to {args.output}")

if __name__ == "__main__":
    main()