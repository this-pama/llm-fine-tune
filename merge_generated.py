#!/usr/bin/env python3
"""
Post-processing script to merge generated chunks per source row.

This script:
1. Post-processes a generated JSONL to group by source_id
2. Sorts by chunk_index within each group
3. Deduplicates repeated phrases
4. Emits merged outputs JSONL with one coherent result per original CSV row

Features:
- Groups chunk results by source_id
- Sorts chunks within each source by chunk_index
- Removes repetitive phrases and content
- Produces final merged output per source row
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Set

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_fine_tune.utils import load_jsonl, write_jsonl


def clean_and_deduplicate_text(text: str) -> str:
    """
    Clean text and remove obvious repetitive patterns.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with reduced repetition
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split into sentences for deduplication
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Deduplicate sentences (case insensitive, fuzzy matching)
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        # Normalize for comparison (lowercase, remove extra spaces)
        normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
        
        # Skip very short sentences
        if len(normalized) < 10:
            continue
        
        # Check for substantial overlap with existing sentences
        is_duplicate = False
        for seen in seen_sentences:
            # Calculate simple similarity (shared words)
            words1 = set(normalized.split())
            words2 = set(seen.split())
            
            if len(words1) > 0:
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                if overlap > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_sentences.append(sentence)
            seen_sentences.add(normalized)
    
    return '. '.join(unique_sentences) + ('.' if unique_sentences else '')


def merge_chunk_summaries(chunk_summaries: List[str]) -> str:
    """
    Merge multiple chunk summaries into a coherent text.
    
    Args:
        chunk_summaries: List of summaries to merge
        
    Returns:
        Merged and deduplicated summary
    """
    if not chunk_summaries:
        return ""
    
    if len(chunk_summaries) == 1:
        return clean_and_deduplicate_text(chunk_summaries[0])
    
    # Combine summaries with section markers
    combined_parts = []
    for i, summary in enumerate(chunk_summaries):
        if summary and not summary.startswith("Error:"):
            cleaned = clean_and_deduplicate_text(summary)
            if cleaned:
                combined_parts.append(cleaned)
    
    if not combined_parts:
        return ""
    
    # Join parts and do final deduplication
    combined_text = ' '.join(combined_parts)
    return clean_and_deduplicate_text(combined_text)


def process_generated_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process generated results to create merged output per source.
    
    Args:
        results: List of generation results from generate_with_olama
        
    Returns:
        List of merged results, one per source_id
    """
    # Separate final summaries from chunk summaries
    final_summaries = {}
    chunk_groups = defaultdict(list)
    
    for result in results:
        source_id = result.get('source_id', 'unknown')
        generation_step = result.get('generation_step', 'unknown')
        
        if generation_step == 'final_summary':
            final_summaries[source_id] = result
        elif generation_step == 'chunk_summary':
            chunk_groups[source_id].append(result)
    
    merged_results = []
    
    # Process each source
    for source_id in set(list(final_summaries.keys()) + list(chunk_groups.keys())):
        print(f"Processing source: {source_id}")
        
        # Get final summary if available
        final_summary_result = final_summaries.get(source_id)
        chunk_results = chunk_groups.get(source_id, [])
        
        # Sort chunk results by chunk_index
        chunk_results.sort(key=lambda x: x.get('chunk_index', 0))
        
        # Extract information
        source_metadata = {}
        chunk_summaries = []
        final_summary = ""
        total_chunks = 0
        
        if final_summary_result:
            source_metadata = final_summary_result.get('source_metadata', {})
            final_summary = final_summary_result.get('final_summary', '')
            total_chunks = final_summary_result.get('total_chunks', 0)
            chunk_summaries = final_summary_result.get('chunk_summaries', [])
        
        # If no final summary but have chunk summaries, merge them
        if not final_summary and chunk_results:
            chunk_summaries = [r.get('chunk_summary', '') for r in chunk_results]
            final_summary = merge_chunk_summaries(chunk_summaries)
            total_chunks = len(chunk_results)
            
            # Extract metadata from first chunk
            if chunk_results:
                first_chunk = chunk_results[0].get('original_chunk', {})
                source_metadata = first_chunk.get('original_row', {})
        
        # Clean final summary
        cleaned_final_summary = clean_and_deduplicate_text(final_summary)
        
        # Create merged result
        merged_result = {
            'source_id': source_id,
            'total_chunks': total_chunks,
            'final_summary': cleaned_final_summary,
            'chunk_summaries': chunk_summaries,
            'source_metadata': source_metadata,
            'processing_step': 'merged'
        }
        
        # Add original row data if available
        if source_metadata:
            merged_result['original_row'] = source_metadata
        
        merged_results.append(merged_result)
    
    return merged_results


def main():
    """Main CLI entrypoint for merging generated results."""
    parser = argparse.ArgumentParser(
        description="Merge generated chunk results per source row with deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge generated results
  python merge_generated.py --input generated_results.jsonl --output merged_results.jsonl
  
  # Process hierarchical generation output
  python merge_generated.py --input hierarchical_output.jsonl --output final_summaries.jsonl
        """
    )
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with generated results")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with merged results")
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from: {input_path}")
    try:
        results = load_jsonl(input_path)
        print(f"Loaded {len(results)} result entries")
    except Exception as e:
        raise SystemExit(f"Error loading input file: {e}")
    
    if not results:
        raise SystemExit("No results found in input file")
    
    # Process and merge results
    print("Processing and merging results...")
    try:
        merged_results = process_generated_results(results)
        
        # Write merged results
        write_jsonl(merged_results, output_path)
        print(f"Successfully wrote {len(merged_results)} merged results to {output_path}")
        
        # Summary stats
        total_chunks = sum(r.get('total_chunks', 0) for r in merged_results)
        successful_summaries = sum(1 for r in merged_results 
                                 if r.get('final_summary') and not r.get('final_summary', '').startswith('Error:'))
        
        print(f"Merged {total_chunks} total chunks into {len(merged_results)} source summaries")
        print(f"Successfully generated {successful_summaries} summaries")
        
    except Exception as e:
        raise SystemExit(f"Error during processing: {e}")


if __name__ == "__main__":
    main()