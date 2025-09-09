#!/usr/bin/env python3
"""
Wrapper script for CSV to JSONL conversion.

This is a convenient wrapper around the src.llm_fine_tune.data_prep module
that provides easy command-line access to CSV to JSONL conversion functionality.

Usage:
  python csv_to_jsonl.py --csv data.csv --output train.jsonl
  python csv_to_jsonl.py --csv data.csv --output train.jsonl --text-column content --filter-empty
"""

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    # Force CSV mode by injecting --csv argument if not present
    if len(sys.argv) > 1 and "--csv" not in " ".join(sys.argv):
        # If user provides --input without --csv, treat it as CSV input
        for i, arg in enumerate(sys.argv):
            if arg == "--input":
                sys.argv[i] = "--csv"
                break
    
    from llm_fine_tune.data_prep import main
    main()