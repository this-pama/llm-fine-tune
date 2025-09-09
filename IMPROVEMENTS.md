# Data Preparation and Generation Improvements

This document demonstrates the new data preparation and hierarchical generation capabilities.

## New Features

### 1. Sentence-Aware Chunking
- Chunks text at sentence boundaries to avoid mid-sentence splits
- Reduces repetition in generated outputs
- Falls back to character-based chunking if no sentence boundaries found

### 2. Metadata Preservation
- Each chunk includes: `source_id`, `chunk_index`, `total_chunks`
- Original row data preserved for stitching results back
- Enables hierarchical summarization workflow

### 3. Hierarchical Generation
- Generate per-chunk summaries first (focused, shorter)
- Merge chunk summaries into final coherent output
- Reduces repetitive content through two-stage approach

### 4. Repetition Reduction
- Added `repetition_penalty=1.1` and `no_repeat_ngram_size=3` to HF generation
- Deduplication in post-processing step
- Better coherence in final outputs

## Usage Examples

### Data Preparation

```bash
# Process single CSV file with sentence-aware chunking
python data_prep.py --input_file data.csv --output chunks.jsonl --max_chars 1500

# Process directory of CSV files
python data_prep.py --input_dir data_inputs --output chunks.jsonl

# Disable chunking (keep full text)
python data_prep.py --input_file data.csv --output full.jsonl --no_chunk

# Custom text column
python data_prep.py --input_file data.csv --output chunks.jsonl --text_column content
```

### Hierarchical Generation

```bash
# Generate with HuggingFace fallback
python generate_with_olama.py --input chunks.jsonl --output results.jsonl --use_hf_fallback --max_gen 150

# Generate with Ollama (when available)
python generate_with_olama.py --input chunks.jsonl --output results.jsonl --model llama3.1 --max_gen 100

# Limit chunk summary length
python generate_with_olama.py --input chunks.jsonl --output results.jsonl --use_hf_fallback --max_gen 50
```

### Post-Processing

```bash
# Merge and deduplicate generated results
python merge_generated.py --input results.jsonl --output final.jsonl
```

## Complete Workflow

```bash
# 1. Prepare data with chunking
python data_prep.py --input_file data.csv --output chunks.jsonl --max_chars 1200

# 2. Generate hierarchical summaries  
python generate_with_olama.py --input chunks.jsonl --output generated.jsonl --use_hf_fallback --max_gen 150

# 3. Merge and clean final results
python merge_generated.py --input generated.jsonl --output final_summaries.jsonl
```

## Output Format

### data_prep.py Output
```json
{
  "source_id": "sample_0",
  "chunk_index": 0,
  "total_chunks": 2,
  "prompt": "Please analyze the following text...",
  "chunk_text": "The actual chunk text...",
  "original_row": {"title": "...", "full_text": "..."}
}
```

### generate_with_olama.py Output
```json
{
  "source_id": "sample_0",
  "total_chunks": 2,
  "chunk_summaries": ["Summary 1", "Summary 2"],
  "final_summary": "Merged coherent summary",
  "source_metadata": {"title": "...", "source_filename": "..."},
  "generation_step": "final_summary"
}
```

### merge_generated.py Output
```json
{
  "source_id": "sample_0",
  "total_chunks": 2,
  "final_summary": "Clean, deduplicated summary",
  "chunk_summaries": ["Summary 1", "Summary 2"],
  "source_metadata": {"title": "...", "source_filename": "..."},
  "processing_step": "merged",
  "original_row": {"title": "...", "full_text": "..."}
}
```

## Benefits

1. **Reduced Repetition**: Sentence-aware chunking + hierarchical generation reduces repeated phrases
2. **Better Coherence**: Two-stage summarization produces more coherent outputs
3. **Metadata Tracking**: Full traceability from original source to final output
4. **Flexible Processing**: Support for single files, directories, chunked or non-chunked processing
5. **Backwards Compatible**: Existing functionality preserved