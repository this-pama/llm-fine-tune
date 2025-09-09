#!/usr/bin/env python3
"""
Data preparation script with chunking support and metadata preservation.

This script reads CSVs from an input directory or a single CSV file, chunks text using 
sentence boundary splitting when available (falling back to character chunks), and emits 
JSONL with fields: source_id, chunk_index, total_chunks, prompt, original_row (metadata).

Features:
- Sentence-aware chunking to reduce mid-sentence splits
- Chunk metadata preservation for stitching results back
- Support for single CSV or directory of CSVs
- Configurable chunking with --no_chunk option
"""

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_fine_tune.utils import chunk_text, clean_text, strip_html, write_jsonl, load_jsonl

# Set CSV field size limit
try:
    csv.field_size_limit(sys.maxsize)
except Exception:
    csv.field_size_limit(10 * 1024 * 1024)


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


def extract_text_from_row(row: Dict[str, str], text_column: str = "full_text") -> str:
    """
    Extract main text content from a CSV row.
    
    Args:
        row: CSV row as dictionary
        text_column: Primary column name to extract text from
        
    Returns:
        Extracted and cleaned text
    """
    # Try specified column first
    if text_column in row and row[text_column]:
        text = clean_text(strip_html(row[text_column]))
        if len(text.strip()) > 10:
            return text
    
    # Fallback to common text column names
    text_candidates = ['full_text', 'text', 'content', 'body', 'description', 'sections']
    for candidate in text_candidates:
        if candidate in row and row[candidate]:
            text = clean_text(strip_html(row[candidate]))
            if len(text.strip()) > 10:
                return text
    
    return ""


def create_prompt_for_chunk(chunk_text: str, row_metadata: Dict[str, str]) -> str:
    """
    Create a prompt for a text chunk with metadata context.
    
    Args:
        chunk_text: The text chunk to process
        row_metadata: Original row metadata for context
        
    Returns:
        Formatted prompt string
    """
    title = row_metadata.get('title', '')
    source_info = row_metadata.get('source_filename', 'unknown')
    
    prompt_parts = [
        "Please analyze the following text and provide a comprehensive summary highlighting the key points, main themes, and important details.",
        ""
    ]
    
    if title:
        prompt_parts.extend([f"Title: {title}", ""])
    
    if source_info != 'unknown':
        prompt_parts.extend([f"Source: {source_info}", ""])
    
    prompt_parts.extend([
        "Text to analyze:",
        chunk_text
    ])
    
    return "\n".join(prompt_parts)


def process_csv_file(csv_path: Path, 
                     text_column: str = "full_text",
                     max_chars: int = 2000,
                     no_chunk: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single CSV file and create chunked entries with metadata.
    
    Args:
        csv_path: Path to CSV file
        text_column: Column containing main text
        max_chars: Maximum characters per chunk
        no_chunk: If True, don't chunk the text
        
    Returns:
        List of processed entries with chunk metadata
    """
    print(f"Processing CSV file: {csv_path}")
    rows = load_csv_rows(csv_path)
    print(f"Found {len(rows)} rows in CSV")
    
    entries = []
    
    for row_idx, row in enumerate(rows):
        # Extract main text
        main_text = extract_text_from_row(row, text_column)
        if not main_text or len(main_text.strip()) < 10:
            print(f"Skipping row {row_idx}: insufficient text content")
            continue
        
        # Generate unique source ID for this row
        source_id = f"{csv_path.stem}_{row_idx}"
        
        # Create row metadata
        row_metadata = {
            'source_filename': csv_path.name,
            'source_id': source_id,
            'row_index': row_idx,
            'title': row.get('title', ''),
            'original_row': row
        }
        
        if no_chunk:
            # Don't chunk - treat entire text as single chunk
            chunks = [main_text]
        else:
            # Chunk text using sentence-aware chunking
            chunks = chunk_text(main_text, max_chars=max_chars, overlap=200, sentence_aware=True)
        
        total_chunks = len(chunks)
        
        # Create entry for each chunk
        for chunk_idx, current_chunk_text in enumerate(chunks):
            prompt = create_prompt_for_chunk(current_chunk_text, row_metadata)
            
            entry = {
                'source_id': source_id,
                'chunk_index': chunk_idx,
                'total_chunks': total_chunks,
                'prompt': prompt,
                'chunk_text': current_chunk_text,  # For debugging/reference
                'original_row': row
            }
            
            entries.append(entry)
    
    print(f"Created {len(entries)} chunk entries from {len(rows)} CSV rows")
    return entries


def process_directory(input_dir: Path,
                      text_column: str = "full_text", 
                      max_chars: int = 2000,
                      no_chunk: bool = False) -> List[Dict[str, Any]]:
    """
    Process all CSV files in a directory.
    
    Args:
        input_dir: Directory containing CSV files
        text_column: Column containing main text
        max_chars: Maximum characters per chunk
        no_chunk: If True, don't chunk the text
        
    Returns:
        List of processed entries from all CSV files
    """
    print(f"Processing directory: {input_dir}")
    
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in directory")
        return []
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_entries = []
    for csv_file in sorted(csv_files):
        entries = process_csv_file(csv_file, text_column, max_chars, no_chunk)
        all_entries.extend(entries)
    
    return all_entries


def main():
    """Main CLI entrypoint for data preparation."""
    parser = argparse.ArgumentParser(
        description="Data preparation: chunk CSV data with metadata preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single CSV file
  python data_prep.py --input_file data.csv --output chunks.jsonl --max_chars 1500
  
  # Process directory of CSV files
  python data_prep.py --input_dir data_inputs --output chunks.jsonl
  
  # Disable chunking (keep full text)
  python data_prep.py --input_file data.csv --output full.jsonl --no_chunk
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str,
                        help="Directory containing CSV files")
    input_group.add_argument("--input_file", type=str,
                        help="Single CSV file path")
    
    # Processing options
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--max_chars", type=int, default=2000,
                        help="Maximum characters per chunk (default: 2000)")
    parser.add_argument("--text_column", type=str, default="full_text",
                        help="Column name containing main text (default: full_text)")
    parser.add_argument("--no_chunk", action="store_true",
                        help="Disable chunking - keep full text as single entry")
    
    args = parser.parse_args()
    
    # Determine input path and mode
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise SystemExit(f"Input file not found: {input_path}")
        if not input_path.suffix.lower() == '.csv':
            raise SystemExit(f"Input file must be a CSV: {input_path}")
    else:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            raise SystemExit(f"Input directory not found: {input_path}")
        if not input_path.is_dir():
            raise SystemExit(f"Input path is not a directory: {input_path}")
    
    output_path = Path(args.output)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process input
        if args.input_file:
            entries = process_csv_file(
                input_path,
                text_column=args.text_column,
                max_chars=args.max_chars,
                no_chunk=args.no_chunk
            )
        else:
            entries = process_directory(
                input_path,
                text_column=args.text_column,
                max_chars=args.max_chars,
                no_chunk=args.no_chunk
            )
        
        if not entries:
            print("No entries created - no valid data found")
            return
        
        # Write output
        write_jsonl(entries, output_path)
        print(f"Successfully wrote {len(entries)} entries to {output_path}")
        
    except Exception as e:
        raise SystemExit(f"Error during processing: {e}")


if __name__ == "__main__":
    main()