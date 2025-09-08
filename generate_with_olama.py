#!/usr/bin/env python3
"""
Generation script - top-level wrapper for src.llm_fine_tune.generate

Usage:
  python generate_with_olama.py --input sft_dataset.jsonl --output sft_dataset.generated.jsonl --max_gen 50
"""

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_fine_tune.generate import main

if __name__ == "__main__":
    main()