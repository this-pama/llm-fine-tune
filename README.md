# LLM Fine-Tune Toolkit

A toolkit for preparing data, generating synthetic outputs, and fine-tuning language models for structured instruction-following tasks.

## Features

- **Data Preparation**: Convert CSV, JSON, JSONL, TXT, and MD files into instruction-style training data
- **Text Generation**: Generate synthetic outputs using Ollama or HuggingFace models with fallback support  
- **Review Pipeline**: Export datasets for human review and curation
- **Fine-tuning**: Train LoRA adapters on large language models
- **Testing & CI**: Comprehensive test suite and GitHub Actions workflows

## Quick Local Test

Get started quickly with the included sample dataset:

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install transformers for HuggingFace fallback
pip install transformers torch
```

### Option 1: Test with HuggingFace Fallback (No Ollama Required)

```bash
# 1. Test data preparation with sample
python data_prep.py --input_dir data --output test_output.jsonl --max_chars 1000

# 2. Generate synthetic outputs using HuggingFace fallback (gpt2)
python generate_with_olama.py \
    --input data/sample_sft_small.jsonl \
    --output test_generated.jsonl \
    --use_hf_fallback \
    --max_gen 3

# 3. Export for review
python export_for_review.py --input test_generated.jsonl --output test_review.csv

# 4. Run tests
pytest tests/ -v
```

### Option 2: Test with Ollama (Requires Ollama Installation)

```bash
# 1. Install and start Ollama
# Visit https://ollama.ai for installation instructions
ollama serve &
ollama pull llama3.1  # or your preferred model

# 2. Generate with Ollama
python generate_with_olama.py \
    --input data/sample_sft_small.jsonl \
    --output test_ollama_generated.jsonl \
    --model llama3.1 \
    --max_gen 3
```

### Option 3: Docker Testing

```bash
# Build and test in container
docker build -t llm-fine-tune .
docker run --rm llm-fine-tune  # Runs tests

# Test generation with HF fallback
docker run --rm llm-fine-tune python generate_with_olama.py \
    --input data/sample_sft_small.jsonl \
    --output /tmp/test.jsonl \
    --use_hf_fallback --max_gen 2
```

## Canonical Metadata Keys

The toolkit uses standardized metadata fields across all scripts:

| Field | Description | Example |
|-------|-------------|---------|
| `source_type` | Document category | `"solution"`, `"experiment"`, `"action_plan"`, `"blog"` |
| `source_filename` | Original file name | `"experiment_platform.csv"` |
| `title` | Document title | `"Water Management Initiative"` |
| `chunk_index` | Chunk number in document | `0`, `1`, `2` |
| `chunk_chars` | Character count of chunk | `1847` |
| `generated_by` | Generation method used | `"ollama"`, `"hf_fallback"` |
| `generated_at` | Generation timestamp | `"2024-01-15 14:30:25"` |
| `review_status` | Manual review status | `"approved"`, `"needs_edit"` |
| `reviewer` | Who reviewed the output | `"john.doe@example.com"` |

## Full Production Workflow

### 1. Data Preparation

Put your source documents in `data_inputs/` and run:

```bash
python data_prep.py --input_dir data_inputs --output sft_dataset.jsonl --max_chars 2000
```

Supports multiple file formats with automatic source type detection based on filename patterns.

### 2. Generate Synthetic Outputs

Choose between Ollama (local) or cloud APIs:

```bash
# Local generation with Ollama
export OLLAMA_HOST='http://localhost:11434'
python generate_with_olama.py \
    --input sft_dataset.jsonl \
    --output sft_dataset.generated.jsonl \
    --model llama3.1 \
    --max_gen 100

# With HuggingFace fallback for reliability
python generate_with_olama.py \
    --input sft_dataset.jsonl \
    --output sft_dataset.generated.jsonl \
    --use_hf_fallback \
    --max_gen 100
```

### 3. Human Review

Export to CSV for spreadsheet review:

```bash
python export_for_review.py \
    --input sft_dataset.generated.jsonl \
    --output review.csv
```

Review in Excel/Google Sheets, then manually update the JSONL with approved examples.

### 4. Fine-tuning

Train LoRA adapters on curated data:

```bash
# For smaller models (local testing)
python train_sft.py \
    --train_file curated_dataset.jsonl \
    --model_name_or_path gpt2 \
    --output_dir ./lora_test

# For production (GPU server)
python train_lora.py \
    --dataset curated_dataset.jsonl \
    --model_name meta-llama/Llama-3-8b \
    --output_dir lora-llama3-8b \
    --epochs 3
```

## Project Structure

```
├── src/llm_fine_tune/       # Main package
│   ├── utils.py             # Core utilities and text processing
│   ├── data_prep.py         # Data preparation module  
│   └── generate.py          # Text generation with fallbacks
├── data/
│   └── sample_sft_small.jsonl  # Sample dataset for testing
├── tests/                   # Test suite
├── data_prep.py            # Top-level script wrappers
├── generate_with_olama.py   # (import from package)
├── export_for_review.py     
├── train_sft.py            
├── train_lora.py           
└── requirements.txt        # Pinned dependencies
```

## Key Improvements

This refactored version includes:

- ✅ **Fixed chunk_text overlap bug** - Proper overlap preservation between text chunks
- ✅ **Robust text extraction** - Handles multiple API response formats (Ollama, OpenAI, HuggingFace)
- ✅ **HuggingFace fallback** - Works without Ollama for testing and CI
- ✅ **Comprehensive tests** - Unit tests and integration tests with pytest
- ✅ **GitHub Actions CI** - Automated linting, testing, and large file detection
- ✅ **Containerization** - Docker support for reproducible environments
- ✅ **Package structure** - Reusable modules with proper imports
- ✅ **No large files** - All datasets moved to external storage with documentation

## Contributing

1. Run tests: `pytest tests/ -v`
2. Check linting: `flake8 src/ --max-line-length=88`
3. Use the sample dataset for testing new features
4. Follow the canonical metadata schema for consistency

## Privacy & Security

- **Local-first**: Use Ollama for sensitive data to avoid external API calls
- **Human review**: Always review generated outputs before training
- **No secrets**: Never commit API keys or sensitive data to the repository
- **External storage**: Large files use Git LFS or external storage solutions