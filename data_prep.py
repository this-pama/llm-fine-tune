#!/usr/bin/env python3
"""
Data preparation script - top-level wrapper for src.llm_fine_tune.data_prep

Usage:
  python data_prep.py --input_dir data_inputs --output sft_dataset.jsonl --max_chars 2000
"""

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_fine_tune.data_prep import main

if __name__ == "__main__":
    main()