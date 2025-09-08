#!/usr/bin/env python3
"""
Generation module with Ollama and HuggingFace fallback support.

Provides robust text generation with retry/backoff for transient errors
and fallback to local HuggingFace models when Ollama is unavailable.
"""

import argparse
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

# Import utilities from package
from .utils import load_jsonl, write_jsonl, build_prompt, extract_text_from_obj


def generate_with_hf_fallback(prompt: str, model_name: str = "gpt2", max_length: int = 200) -> str:
    """
    Generate text using a small HuggingFace model as fallback.
    
    Args:
        prompt: Input prompt
        model_name: HF model name (default: gpt2 for minimal size)
        max_length: Maximum generation length
        
    Returns:
        Generated text
    """
    try:
        from transformers import pipeline, set_seed
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create generation pipeline
        generator = pipeline('text-generation', model=model_name, 
                           device_map='auto' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
        
        # Generate with controlled parameters
        outputs = generator(
            prompt,
            max_length=min(len(prompt.split()) + max_length, 1024),
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        if outputs and len(outputs) > 0:
            generated = outputs[0]['generated_text']
            # Remove the original prompt from the output
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            return generated
        
    except ImportError:
        return "Error: transformers package not available for HF fallback"
    except Exception as e:
        return f"Error in HF generation: {e}"
    
    return "Error: No output generated"


def generate_with_ollama(conversation: List[Dict[str, str]], model: str, 
                        stream_output: bool = False,
                        options: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate response using Ollama with error handling.
    
    Args:
        conversation: List of message dicts with 'role' and 'content'
        model: Ollama model name
        stream_output: Whether to stream output
        options: Optional generation parameters
        
    Returns:
        Generated text string
    """
    try:
        import ollama
        
        # Get Ollama host from environment or use default
        host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        client = ollama.Client(host=host)
        
        opts = options or {}
        
        response = client.chat(
            model=model,
            messages=conversation,
            stream=stream_output,
            options=opts
        )
        
        if stream_output:
            # Handle streaming response
            result_parts = []
            for chunk in response:
                if 'message' in chunk:
                    content = chunk['message'].get('content', '')
                    if content:
                        result_parts.append(content)
            return ''.join(result_parts)
        else:
            # Handle single response
            return extract_text_from_obj(response) or ""
            
    except ImportError:
        raise RuntimeError("ollama package not available")
    except Exception as e:
        raise RuntimeError(f"Ollama generation failed: {e}")


def generate_with_retry(prompt: str, model: str = "llama3.1", 
                       use_hf_fallback: bool = False,
                       max_retries: int = 3, base_delay: float = 1.0) -> str:
    """
    Generate text with retry logic and optional HF fallback.
    
    Args:
        prompt: Input prompt
        model: Model name for Ollama
        use_hf_fallback: Whether to use HF fallback if Ollama fails
        max_retries: Maximum retry attempts for transient errors
        base_delay: Base delay for exponential backoff
        
    Returns:
        Generated text
    """
    conversation = [{"role": "user", "content": prompt}]
    
    # Try Ollama first (unless explicitly using HF fallback)
    if not use_hf_fallback:
        for attempt in range(max_retries):
            try:
                return generate_with_ollama(conversation, model)
            except RuntimeError as e:
                error_msg = str(e).lower()
                # Check for transient errors
                if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'temporary']):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Transient error, retrying in {delay:.1f}s: {e}")
                        time.sleep(delay)
                        continue
                
                # For non-transient errors or final attempt, fall back if enabled
                if use_hf_fallback:
                    print(f"Ollama failed, falling back to HuggingFace: {e}")
                    break
                else:
                    raise
    
    # Use HuggingFace fallback
    if use_hf_fallback:
        return generate_with_hf_fallback(prompt, model_name="gpt2", max_length=200)
    
    return "Error: Generation failed and no fallback available"


def main():
    """Main CLI for generation with Ollama/HF fallback."""
    parser = argparse.ArgumentParser(description="Generate text completions using Ollama or HF fallback")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with examples")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file")
    parser.add_argument("--model", type=str, default="llama3.1",
                        help="Model name for Ollama")
    parser.add_argument("--max_gen", type=int, default=50,
                        help="Maximum number of examples to generate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--use_hf_fallback", action="store_true",
                        help="Use HuggingFace fallback if Ollama unavailable")
    parser.add_argument("--stream", type=str, default="false",
                        help="Enable streaming output (true/false)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    
    data = load_jsonl(input_path)
    to_gen_idxs = [i for i, item in enumerate(data) if not item.get("output")]
    
    if not to_gen_idxs:
        print("No empty-output examples found. Nothing to do.")
        return
    
    max_gen = min(len(to_gen_idxs), args.max_gen)
    print(f"{len(to_gen_idxs)} empty-output examples found; will generate up to {max_gen}")
    
    stream_flag = args.stream.lower() == "true"
    generated = 0
    start_time = time.time()
    
    # Process in batches
    for base in range(0, max_gen, args.batch_size):
        batch = to_gen_idxs[base: base + args.batch_size]
        for idx in batch:
            example = data[idx]
            prompt = build_prompt(example)
            
            print(f"Generating {generated + 1}/{max_gen}...")
            
            try:
                if args.use_hf_fallback:
                    # Use HF fallback directly
                    output = generate_with_retry(prompt, args.model, use_hf_fallback=True)
                else:
                    # Try Ollama first
                    conversation = [{"role": "user", "content": prompt}]
                    options = {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "stop": ["### Instruction:", "### Input:"]
                    }
                    output = generate_with_ollama(conversation, args.model, stream_flag, options)
                
                if output and output.strip():
                    data[idx]["output"] = output.strip()
                    # Add generation metadata
                    if "metadata" not in data[idx]:
                        data[idx]["metadata"] = {}
                    data[idx]["metadata"]["generated_by"] = "hf_fallback" if args.use_hf_fallback else "ollama"
                    data[idx]["metadata"]["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    generated += 1
                else:
                    print(f"Warning: Empty output for example {idx}")
                    
            except Exception as e:
                print(f"Error generating for example {idx}: {e}")
                if args.use_hf_fallback:
                    # Fallback to HF
                    try:
                        output = generate_with_hf_fallback(prompt)
                        if output:
                            data[idx]["output"] = output.strip()
                            generated += 1
                    except Exception as e2:
                        print(f"HF fallback also failed: {e2}")
    
    # Write results
    write_jsonl(data, Path(args.output))
    elapsed = time.time() - start_time
    print(f"Generated {generated} outputs in {elapsed:.1f}s")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
