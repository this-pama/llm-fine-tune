#!/usr/bin/env python3
"""
Generation script using Ollama (or HuggingFace fallback) with hierarchical summarization.

This script:
1. Uses the new data_prep JSONL format as input to generate per-chunk summaries
2. For each source_id, generates a short summary per chunk (max_gen tokens)
3. After generating chunk summaries, joins them and runs a final summarization call 
   to produce a single output per source row
4. Supports CLI options for model selection and generation parameters

Features:
- Hierarchical summarization (chunk-level then merge to final output)
- Chunk metadata preservation and tracking
- Configurable generation parameters for repetition reduction
- HuggingFace fallback support
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_fine_tune.utils import load_jsonl, write_jsonl
from llm_fine_tune.generate import generate_with_hf_fallback, summarize_text


def generate_chunk_summary(chunk_entry: Dict[str, Any], 
                          model: str = "llama3.1",
                          use_hf_fallback: bool = False,
                          hf_fallback_model: Optional[str] = None,
                          max_gen: int = 200) -> str:
    """
    Generate a summary for a single chunk entry.
    
    Args:
        chunk_entry: Chunk entry from data_prep output
        model: Model name for generation
        use_hf_fallback: Whether to use HuggingFace fallback
        hf_fallback_model: Specific HF model to use
        max_gen: Maximum tokens to generate
        
    Returns:
        Generated summary text
    """
    prompt = chunk_entry.get('prompt', '')
    chunk_text = chunk_entry.get('chunk_text', '')
    
    if not prompt and chunk_text:
        # Fallback prompt if none provided
        prompt = f"Summarize the following text concisely:\n\n{chunk_text}"
    
    if use_hf_fallback:
        # Use HuggingFace with repetition penalty
        return generate_with_hf_fallback(
            prompt, 
            model_name=hf_fallback_model, 
            max_new_tokens=max_gen
        )
    else:
        # TODO: Implement Ollama generation call
        # For now, use HF fallback as placeholder
        print(f"Warning: Ollama generation not implemented, using HF fallback")
        return generate_with_hf_fallback(
            prompt, 
            model_name=hf_fallback_model, 
            max_new_tokens=max_gen
        )


def generate_final_summary(chunk_summaries: List[str],
                          source_metadata: Dict[str, Any],
                          use_hf_fallback: bool = False,
                          hf_fallback_model: Optional[str] = None,
                          max_gen: int = 300) -> str:
    """
    Generate a final summary from chunk summaries.
    
    Args:
        chunk_summaries: List of chunk summaries to merge
        source_metadata: Metadata from original source
        use_hf_fallback: Whether to use HuggingFace fallback
        hf_fallback_model: Specific HF model to use
        max_gen: Maximum tokens to generate
        
    Returns:
        Final merged summary
    """
    # Combine chunk summaries
    combined_text = "\n\n".join(f"Section {i+1}: {summary}" 
                                for i, summary in enumerate(chunk_summaries))
    
    # Create final summarization prompt
    title = source_metadata.get('title', '')
    source_filename = source_metadata.get('source_filename', '')
    
    prompt_parts = [
        "Create a comprehensive summary by merging the following section summaries into a coherent, unified analysis.",
        "Focus on key themes, main points, and important details while avoiding repetition.",
        ""
    ]
    
    if title:
        prompt_parts.extend([f"Document Title: {title}", ""])
    
    if source_filename:
        prompt_parts.extend([f"Source: {source_filename}", ""])
    
    prompt_parts.extend([
        "Section summaries to merge:",
        combined_text,
        "",
        "Unified Summary:"
    ])
    
    final_prompt = "\n".join(prompt_parts)
    
    if use_hf_fallback:
        return generate_with_hf_fallback(
            final_prompt,
            model_name=hf_fallback_model,
            max_new_tokens=max_gen
        )
    else:
        # TODO: Implement Ollama generation call
        print(f"Warning: Ollama generation not implemented, using HF fallback")
        return generate_with_hf_fallback(
            final_prompt,
            model_name=hf_fallback_model,
            max_new_tokens=max_gen
        )


def process_chunks_hierarchically(chunks: List[Dict[str, Any]],
                                 model: str = "llama3.1",
                                 use_hf_fallback: bool = False,
                                 hf_fallback_model: Optional[str] = None,
                                 max_gen: int = 200) -> List[Dict[str, Any]]:
    """
    Process chunks using hierarchical summarization.
    
    Args:
        chunks: List of chunk entries from data_prep
        model: Model name for generation
        use_hf_fallback: Whether to use HuggingFace fallback
        hf_fallback_model: Specific HF model to use
        max_gen: Maximum tokens per chunk summary
        
    Returns:
        List of processed entries with hierarchical summaries
    """
    # Group chunks by source_id
    source_groups = defaultdict(list)
    for chunk in chunks:
        source_id = chunk.get('source_id', 'unknown')
        source_groups[source_id].append(chunk)
    
    results = []
    
    for source_id, source_chunks in source_groups.items():
        print(f"\nProcessing source: {source_id} ({len(source_chunks)} chunks)")
        
        # Sort chunks by chunk_index
        source_chunks.sort(key=lambda x: x.get('chunk_index', 0))
        
        # Generate summary for each chunk
        chunk_summaries = []
        chunk_results = []
        
        for chunk in source_chunks:
            chunk_idx = chunk.get('chunk_index', 0)
            print(f"  Generating summary for chunk {chunk_idx}...")
            
            try:
                summary = generate_chunk_summary(
                    chunk, model, use_hf_fallback, hf_fallback_model, max_gen
                )
                chunk_summaries.append(summary)
                
                # Store chunk result with summary
                chunk_result = {
                    'source_id': source_id,
                    'chunk_index': chunk_idx,
                    'total_chunks': chunk.get('total_chunks', len(source_chunks)),
                    'chunk_summary': summary,
                    'original_chunk': chunk,
                    'generation_step': 'chunk_summary'
                }
                chunk_results.append(chunk_result)
                
            except Exception as e:
                print(f"    Error generating chunk summary: {e}")
                chunk_summaries.append(f"Error: {e}")
                chunk_results.append({
                    'source_id': source_id,
                    'chunk_index': chunk_idx,
                    'chunk_summary': f"Error: {e}",
                    'original_chunk': chunk,
                    'generation_step': 'chunk_summary'
                })
        
        # Generate final summary from chunk summaries
        if chunk_summaries and any(not s.startswith("Error:") for s in chunk_summaries):
            print(f"  Generating final summary for {source_id}...")
            
            # Get metadata from first chunk
            source_metadata = source_chunks[0].get('original_row', {})
            source_metadata.update({
                'source_filename': source_chunks[0].get('original_row', {}).get('source_filename', ''),
                'title': source_chunks[0].get('original_row', {}).get('title', '')
            })
            
            try:
                final_summary = generate_final_summary(
                    [s for s in chunk_summaries if not s.startswith("Error:")],
                    source_metadata,
                    use_hf_fallback,
                    hf_fallback_model,
                    max_gen * 2  # Allow more tokens for final summary
                )
                
                # Create final result entry
                final_result = {
                    'source_id': source_id,
                    'total_chunks': len(source_chunks),
                    'chunk_summaries': chunk_summaries,
                    'final_summary': final_summary,
                    'source_metadata': source_metadata,
                    'generation_step': 'final_summary'
                }
                results.append(final_result)
                
            except Exception as e:
                print(f"    Error generating final summary: {e}")
                final_result = {
                    'source_id': source_id,
                    'total_chunks': len(source_chunks),
                    'chunk_summaries': chunk_summaries,
                    'final_summary': f"Error: {e}",
                    'source_metadata': source_metadata,
                    'generation_step': 'final_summary'
                }
                results.append(final_result)
        
        # Add individual chunk results too
        results.extend(chunk_results)
    
    return results


def main():
    """Main CLI entrypoint for hierarchical generation."""
    parser = argparse.ArgumentParser(
        description="Generate summaries using hierarchical approach with chunked data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with Ollama (when implemented)
  python generate_with_olama.py --input chunks.jsonl --output results.jsonl --model llama3.1 --max_gen 150
  
  # Generate with HuggingFace fallback
  python generate_with_olama.py --input chunks.jsonl --output results.jsonl --use_hf_fallback --hf_fallback_model gpt2 --max_gen 100
  
  # Limit generation length
  python generate_with_olama.py --input chunks.jsonl --output results.jsonl --use_hf_fallback --max_gen 50
        """
    )
    
    # Input/output
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file from data_prep")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with generated summaries")
    
    # Model options
    parser.add_argument("--model", type=str, default="llama3.1",
                        help="Ollama model name (default: llama3.1)")
    parser.add_argument("--use_hf_fallback", action="store_true",
                        help="Use HuggingFace instead of Ollama")
    parser.add_argument("--hf_fallback_model", type=str,
                        help="Specific HuggingFace model to use")
    
    # Generation parameters
    parser.add_argument("--max_gen", type=int, default=200,
                        help="Maximum tokens per chunk summary (default: 200)")
    parser.add_argument("--max_chars", type=int, 
                        help="Legacy parameter (ignored - use max_gen)")
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    print(f"Loading chunks from: {input_path}")
    try:
        chunks = load_jsonl(input_path)
        print(f"Loaded {len(chunks)} chunk entries")
    except Exception as e:
        raise SystemExit(f"Error loading input file: {e}")
    
    if not chunks:
        raise SystemExit("No chunks found in input file")
    
    # Process chunks hierarchically
    print(f"\nStarting hierarchical generation...")
    print(f"Model: {args.model}")
    print(f"Use HF fallback: {args.use_hf_fallback}")
    print(f"HF model: {args.hf_fallback_model}")
    print(f"Max tokens per chunk: {args.max_gen}")
    
    try:
        results = process_chunks_hierarchically(
            chunks,
            model=args.model,
            use_hf_fallback=args.use_hf_fallback,
            hf_fallback_model=args.hf_fallback_model,
            max_gen=args.max_gen
        )
        
        # Write results
        write_jsonl(results, output_path)
        print(f"\nSuccessfully wrote {len(results)} results to {output_path}")
        
        # Summary stats
        final_summaries = [r for r in results if r.get('generation_step') == 'final_summary']
        chunk_summaries = [r for r in results if r.get('generation_step') == 'chunk_summary']
        
        print(f"Generated {len(final_summaries)} final summaries from {len(chunk_summaries)} chunk summaries")
        
    except Exception as e:
        raise SystemExit(f"Error during generation: {e}")


if __name__ == "__main__":
    main()