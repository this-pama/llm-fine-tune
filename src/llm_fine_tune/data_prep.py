#!/usr/bin/env python3
"""
Data preparation module for creating instruction-style JSONL from document sources.

Handles CSV, JSON, JSONL, TXT, and MD files, applying different instruction templates
based on source type detection.
"""

import argparse
import json
import csv
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

# Import utilities from package
from .utils import chunk_text, clean_text, strip_html, write_jsonl

# Set CSV field size limit
try:
    csv.field_size_limit(sys.maxsize)
except Exception:
    csv.field_size_limit(10 * 1024 * 1024)


# ---- Source type detection ----
def infer_source_type(filename: str) -> str:
    """Infer document source type from filename patterns."""
    fname_lower = filename.lower()
    if "solution" in fname_lower:
        return "solution"
    elif "experiment" in fname_lower:
        return "experiment"
    elif "action" in fname_lower or "plan" in fname_lower:
        return "action_plan"
    elif "blog" in fname_lower:
        return "blog"
    else:
        return "unknown"


# ---- Instruction templates for different source types ----
INSTRUCTION_TEMPLATES = {
    "solution": (
        "You are an expert analyst reviewing innovative solutions."
        " Provide: 'Brief solution description:', 'Methodology or approach:', 'Key results / impact achieved:', "
        "'Success factors / enablers:', 'Challenges encountered:', 'Transferability / replication potential:', "
        "and 'Recommendations for improvement or scaling:'."
    ),
    "experiment": (
        "You are a research analyst evaluating innovation experiments."
        " Outline: 'Hypothesis or research question:', 'Methodology and scope:', 'Key findings / evidence:', "
        "'Insights and learnings:', 'Limitations or considerations:', 'Implications for policy or practice:', "
        "and 'Next steps or further research suggestions:'."
    ),
    "action_plan": (
        "You are a practical advisor on implementing action plans."
        " Summarize: 'Action goal:', 'Learning questions:', 'Planned steps / timeline:', 'Resources needed / partners:', "
        "'Outcomes achieved (if noted):', 'Lessons learned:', and 'Follow-up actions / monitoring suggestions:'."
    ),
    "blog": (
        "You are a reflective analyst synthesizing blog and publication insights."
        " Produce a concise 'Summary of reflection:', 'Key lessons / insights:', 'Policy or practice implications:', "
        "and 'Suggested further reading or actions:'."
    ),
    "unknown": (
        "You are a helpful assistant. Summarize the document focusing on key points and recommended actions."
    )
}


# ---- File loaders ----
def load_text_file(path: Path) -> str:
    """Load a text file."""
    return path.read_text(encoding="utf-8")


def load_json(path: Path):
    """Load JSON file, returning list if possible."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Return list if top-level contains a list field
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    return data


def load_csv_simple(path: Path) -> List[dict]:
    """Read CSV robustly with fallback to pandas if needed."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return list(reader)
    except Exception as e:
        print(f"Warning: CSV read failed with {e}, trying pandas fallback")
        try:
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict("records")
        except ImportError:
            raise SystemExit(f"Failed to read {path}: {e}. Install pandas for robust CSV reading.")


# ---- Text extraction for various record shapes ----
def _extract_from_sections_field(val: Union[str, list, dict]) -> List[str]:
    """
    Extract text from nested 'sections' field that may contain JSON.
    """
    bits = []

    def rec(node):
        if node is None:
            return
        if isinstance(node, str):
            s = node.strip()
            # Detect JSON strings
            if (s.startswith("{") or s.startswith("[")) and len(s) > 50:
                try:
                    parsed = json.loads(s)
                    rec(parsed)
                    return
                except Exception:
                    pass
            # Plain string
            cleaned = strip_html(s)
            if cleaned.strip():
                bits.append(cleaned)
            return
                
        if isinstance(node, dict):
            # Common text keys
            for k in ("txt", "text", "title", "html", "body", "content", "lead", "instruction"):
                if k in node and node[k]:
                    rec(node[k])
            # Recurse into other dict values
            for k, v in node.items():
                if k not in ("txt", "text", "title", "html", "body", "content", "lead", "instruction"):
                    rec(v)
        elif isinstance(node, list):
            for item in node:
                rec(item)

    rec(val)
    return bits


def extract_text_from_record(record: Dict[str, Any]) -> str:
    """Extract text from a record (CSV row or JSON object)."""
    # 1) Try 'full_text' first
    if "full_text" in record:
        full_text = record["full_text"]
        if isinstance(full_text, str) and len(full_text.strip()) > 20:
            return clean_text(strip_html(full_text))

    # 2) Try 'sections' field with JSON parsing
    if "sections" in record:
        sec_val = record["sections"]
        try:
            if isinstance(sec_val, str):
                sec_val_stripped = sec_val.strip()
                if sec_val_stripped.startswith("[") or sec_val_stripped.startswith("{"):
                    parsed = json.loads(sec_val_stripped)
                    bits = _extract_from_sections_field(parsed)
                    if bits:
                        return clean_text("\n\n".join(b.strip() for b in bits if b.strip()))
            else:
                bits = _extract_from_sections_field(sec_val)
                if bits:
                    return clean_text("\n\n".join(b.strip() for b in bits if b.strip()))
        except Exception:
            # Fallback to treating sections as plain text
            cleaned = strip_html(str(sec_val))
            if len(cleaned.strip()) > 20:
                return clean_text(cleaned)

    # 3) Fallback: check other candidate fields
    pieces = []
    for k, v in record.items():
        if isinstance(v, str) and len(v.strip()) > 40:
            cleaned = strip_html(v).strip()
            if len(cleaned) > 40:
                pieces.append(cleaned)

    if pieces:
        # Take the longest piece
        longest = max(pieces, key=len)
        return clean_text(longest)

    return ""


# ---- Example creation ----
def create_examples_from_doc(source_filename: str, title: str, text: str,
                             max_chars: int = 2000) -> List[Dict]:
    """Create SFT examples from a document."""
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
            "output": "",  # Empty for manual annotation or synthetic generation
            "metadata": {
                "source_type": source_type,
                "source_filename": source_filename,
                "title": title or "",
                "chunk_index": i,
                "chunk_chars": len(c)
            }
        })
    return examples


# ---- Document collection ----
def collect_docs_from_dir(input_dir: Path) -> List[Dict]:
    """Collect documents from directory, handling various file formats."""
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
                            if text and len(text) > 20:
                                title = it.get("title") or it.get("name") or ""
                                docs.append({"file": p.name, "title": title, "text": clean_text(text)})
            elif p.suffix.lower() == ".csv":
                rows = load_csv_simple(p)
                for r in rows:
                    text = extract_text_from_record(r)
                    if text and len(text) > 20:
                        title = r.get("title") or r.get("name") or ""
                        docs.append({"file": p.name, "title": title, "text": clean_text(text)})
            elif p.suffix.lower() in (".md", ".txt"):
                text = load_text_file(p)
                if text and len(text) > 20:
                    docs.append({"file": p.name, "title": p.stem, "text": clean_text(text)})
            else:
                # Try as text file
                text = load_text_file(p)
                if text and len(text) > 20:
                    docs.append({"file": p.name, "title": p.stem, "text": clean_text(text)})
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    return docs


# ---- CSV to JSONL specific functionality ----

# Default instruction template for CSV data
CSV_DEFAULT_INSTRUCTION = (
    "Please analyze the following text and provide a comprehensive summary "
    "highlighting the key points, main themes, and important details."
)


def extract_title_from_text(text: str, max_length: int = 100) -> str:
    """
    Extract a title from the beginning of text using conservative heuristics.
    
    Args:
        text: The text to extract title from
        max_length: Maximum length of extracted title
        
    Returns:
        Extracted title or empty string if none found
    """
    if not text or not text.strip():
        return ""
    
    # Clean the text first
    text = text.strip()
    
    # Try to find a title in the first few lines
    lines = text.split('\n')[:3]  # Look at first 3 lines
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove common markup/formatting
        line = re.sub(r'^#+\s*', '', line)  # Remove markdown headers
        line = re.sub(r'^-+\s*', '', line)  # Remove dashes
        line = re.sub(r'^\*+\s*', '', line)  # Remove asterisks
        line = strip_html(line)
        line = line.strip()
        
        # Check if line looks like a title (reasonable length, not too short/long)
        if 10 <= len(line) <= max_length and not line.endswith('.'):
            # Avoid lines that look like URLs, emails, or code
            if not re.search(r'(https?://|www\.|@|<|>|\{|\})', line):
                return line
    
    # Fallback: use first sentence up to max_length
    first_sentence = text.split('.')[0].strip()
    if 10 <= len(first_sentence) <= max_length:
        return first_sentence
    
    # Last resort: truncate beginning
    if len(text) > max_length:
        return text[:max_length].strip() + "..."
    
    return text.strip()


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load CSV file and return list of dictionaries.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of row dictionaries
    """
    rows = []
    try:
        with csv_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert all values to strings and strip whitespace
                cleaned_row = {}
                for k, v in row.items():
                    if k is not None:  # Skip None keys
                        cleaned_row[k.strip()] = str(v).strip() if v is not None else ""
                rows.append(cleaned_row)
    except Exception as e:
        # Try with different encoding
        try:
            with csv_path.open('r', encoding='latin-1', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned_row = {}
                    for k, v in row.items():
                        if k is not None:
                            cleaned_row[k.strip()] = str(v).strip() if v is not None else ""
                    rows.append(cleaned_row)
        except Exception as e2:
            raise SystemExit(f"Failed to read CSV {csv_path}: {e} (also tried latin-1: {e2})")
    
    return rows


def csv_row_to_jsonl_entry(row: Dict[str, str], 
                          text_column: str = "full_text",
                          instruction: str = CSV_DEFAULT_INSTRUCTION) -> Optional[Dict[str, str]]:
    """
    Convert a CSV row to a JSONL entry for SFT training.
    
    Args:
        row: Dictionary representing a CSV row
        text_column: Name of column containing the main text
        instruction: Instruction text to use
        
    Returns:
        Dictionary with instruction/input/output fields, or None if no valid text
    """
    # Ignore 'section' column entirely as specified
    if 'section' in row:
        row = {k: v for k, v in row.items() if k != 'section'}
    
    # Extract main text from specified column
    main_text = ""
    if text_column in row and row[text_column]:
        main_text = clean_text(strip_html(row[text_column]))
    else:
        # Fallback: look for common text column names
        text_candidates = ['full_text', 'text', 'content', 'body', 'description']
        for candidate in text_candidates:
            if candidate in row and row[candidate]:
                main_text = clean_text(strip_html(row[candidate]))
                break
    
    if not main_text or len(main_text.strip()) < 10:
        return None  # Skip rows without sufficient text
    
    # Extract or generate title
    title = ""
    if 'title' in row and row['title']:
        title = clean_text(strip_html(row['title']))
    else:
        title = extract_title_from_text(main_text)
    
    # Look for explicit output/label fields
    output = ""
    output_candidates = ['output', 'label', 'target', 'summary', 'response', 'answer']
    for candidate in output_candidates:
        if candidate in row and row[candidate]:
            output = clean_text(strip_html(row[candidate]))
            break
    
    # Construct input field
    input_parts = []
    if title:
        input_parts.append(f"Title: {title}")
    input_parts.append(f"Text: {main_text}")
    
    input_text = "\n\n".join(input_parts)
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output  # May be empty if no explicit output field found
    }


def process_csv_to_jsonl(csv_path: Path, 
                        output_path: Path,
                        text_column: str = "full_text",
                        instruction: Optional[str] = None,
                        filter_empty_output: bool = False) -> int:
    """
    Process a CSV file and convert it to JSONL format for SFT training.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSONL file
        text_column: Name of column containing main text
        instruction: Custom instruction text (uses default if None)
        filter_empty_output: If True, skip entries with empty output
        
    Returns:
        Number of entries written
    """
    if instruction is None:
        instruction = CSV_DEFAULT_INSTRUCTION
    
    print(f"Reading CSV from: {csv_path}")
    rows = load_csv_rows(csv_path)
    print(f"Found {len(rows)} rows in CSV")
    
    entries = []
    skipped = 0
    
    for i, row in enumerate(rows):
        entry = csv_row_to_jsonl_entry(row, text_column=text_column, instruction=instruction)
        
        if entry is None:
            skipped += 1
            continue
            
        if filter_empty_output and not entry["output"]:
            skipped += 1
            continue
            
        entries.append(entry)
    
    print(f"Converted {len(entries)} entries (skipped {skipped})")
    
    if entries:
        write_jsonl(entries, output_path)
        print(f"Wrote JSONL to: {output_path}")
    else:
        print("No valid entries to write")
    
    return len(entries)


# ---- CLI entrypoint ----
def main():
    """Main CLI entrypoint for data preparation."""
    parser = argparse.ArgumentParser(
        description="Data preparation: convert documents to instruction-style JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mode 1 - Directory processing (original functionality):
  python -m llm_fine_tune.data_prep --input_dir data_inputs --output sft_dataset.jsonl

Mode 2 - CSV to JSONL conversion (new functionality):
  python -m llm_fine_tune.data_prep --csv input.csv --output train.jsonl
  python -m llm_fine_tune.data_prep --csv input.csv --output train.jsonl --text-column content --filter-empty
        """
    )
    
    # Original functionality arguments
    parser.add_argument("--input_dir", type=str, default="data_inputs",
                        help="Directory containing source documents (original mode)")
    parser.add_argument("--max_chars", type=int, default=2000,
                        help="Maximum characters per chunk (original mode)")
    
    # New CSV functionality arguments
    parser.add_argument("--csv", type=str,
                        help="Input CSV file path (CSV mode)")
    parser.add_argument("--text-column", type=str, default="full_text",
                        help="Name of column containing main text (CSV mode)")
    parser.add_argument("--instruction", type=str,
                        help="Custom instruction text (CSV mode)")
    parser.add_argument("--filter-empty", action="store_true",
                        help="Filter out entries with empty output field (CSV mode)")
    
    # Common arguments
    parser.add_argument("--output", type=str, default="sft_dataset.jsonl",
                        help="Output JSONL file")
    
    args = parser.parse_args()

    if args.csv:
        # CSV to JSONL mode
        csv_path = Path(args.csv)
        output_path = Path(args.output)
        
        if not csv_path.exists():
            raise SystemExit(f"Input CSV file not found: {csv_path}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            count = process_csv_to_jsonl(
                csv_path=csv_path,
                output_path=output_path,
                text_column=args.text_column,
                instruction=args.instruction,
                filter_empty_output=args.filter_empty
            )
            print(f"Successfully processed {count} entries")
        except Exception as e:
            raise SystemExit(f"Error processing CSV: {e}")
    else:
        # Original directory processing mode
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
