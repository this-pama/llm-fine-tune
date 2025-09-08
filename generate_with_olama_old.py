#!/usr/bin/env python3
"""
Minimal generator that calls Ollama via the Python client (ollama_client.chat).

- No fallbacks (no CLI/HTTP/embeddings).
- Uses the exact client style you provided (ollama_client.chat).
- By default streaming is disabled so the client should return a single JSON
  response object (no NDJSON chunks). If you want streaming set --stream.
- Writes only the cleaned textual response into each example's "output" field.

Usage example:
  export OLAMA_MODEL_NAME="llama3.2"
  python generate_with_olama.py --input sft_dataset.jsonl --output sft_dataset.generated.jsonl \
    --max_gen 10 --batch_size 2 --max_new_tokens 512 --temperature 0.1 --stream false --model_name llama3.2

Note: install the Python client package you use for Ollama so that
from ollama_client import ollama_client works in your venv.
"""
from pathlib import Path
import argparse
import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import os
from ollama import Client

# Get the Ollama host from the environment variable or default to localhost
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# Initialize the Ollama client
ollama_client = Client(host=OLLAMA_HOST)  # Use 'host' instead of 'base_url'

PROMPT_WRAPPER = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items: List[Dict[str, Any]], path: Path):
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def build_prompt(example: Dict[str, Any]) -> str:
    return PROMPT_WRAPPER.format(instruction=example.get("instruction", ""), input=example.get("input", ""))


def extract_text_from_obj(obj: Any) -> Optional[str]:
    """
    Extract model text from various possible response shapes:
    - {"response": "..." }
    - {"message": {"content": "..." } }
    - {"text": "..."} or {"result": "..."} etc.
    - plain string
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # common keys
        for k in ("response", "text", "result", "generated_text", "output", "content"):
            if k in obj and isinstance(obj[k], str) and obj[k]:
                return obj[k]
        if "message" in obj and isinstance(obj["message"], dict):
            c = obj["message"].get("content")
            if isinstance(c, str) and c:
                return c
        # nested generations / choices
        if "generations" in obj and isinstance(obj["generations"], list) and obj["generations"]:
            first = obj["generations"][0]
            if isinstance(first, dict):
                for k in ("text", "content"):
                    if k in first and isinstance(first[k], str) and first[k]:
                        return first[k]
        if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
            ch = obj["choices"][0]
            if isinstance(ch, dict):
                if "text" in ch and isinstance(ch["text"], str):
                    return ch["text"]
                if "message" in ch and isinstance(ch["message"], dict) and "content" in ch["message"]:
                    return ch["message"]["content"]
    return None


def assistant_response_from_ollama(conversation: List[Dict[str, str]],
                                   model: str,
                                   stream_output: bool,
                                   options: Optional[Dict[str, Any]] = None) -> str:
    """
    Call ollama_client.chat and return a single cleaned string result.
    If stream_output is False the client should return a single object.
    If it returns an iterator (stream), we'll concatenate textual parts in order.
    """
    opts = options or {}

    stream = ollama_client.chat(
        model=model,
        options=opts,
        messages=conversation,
        stream=stream_output,
    )

    if stream.message:
        return stream.message.content
    else:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file to write generated results")
    parser.add_argument("--model_name", required=True, help="Ollama model name (e.g., llama3.2)")
    parser.add_argument("--max_gen", type=int, default=None, help="Max number of examples to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (requests per loop)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", type=str, choices=("true", "false"), default="false",
                        help="Whether to stream responses from Ollama. Default false (single JSON response).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    data = load_jsonl(input_path)
    to_gen_idxs = [i for i, it in enumerate(data) if not it.get("output")]
    if not to_gen_idxs:
        print("No empty-output examples found. Nothing to do.")
        return

    if args.max_gen is None:
        max_gen = len(to_gen_idxs)
    else:
        max_gen = min(len(to_gen_idxs), args.max_gen)
    print(f"{len(to_gen_idxs)} empty-output examples found; will generate up to {max_gen}")

    stream_flag = True if args.stream.lower() == "true" else False

    generated = 0
    start = time.time()
    for base in range(0, max_gen, args.batch_size):
        batch = to_gen_idxs[base: base + args.batch_size]
        for idx in batch:
            example = data[idx]
            prompt = build_prompt(example)
            conversation = [{"role": "user", "content": prompt}]
            options = {
                "temperature": float(args.temperature),
                "max_new_tokens": int(args.max_new_tokens),
                # keep a default seed if desired, user can edit script to tune
                "seed": 42,
            }
            try:
                text = assistant_response_from_ollama(conversation=conversation, model=args.model_name, stream_output=stream_flag, options=options)
            except Exception as e:
                print(f"Generation failed for index={idx}: {e}", file=sys.stderr)
                text = ""
            # Set output (cleaned text)
            data[idx]["output"] = text
            # Minimal metadata
            meta = data[idx].get("metadata", {})
            meta.update({
                "generated_by": args.model_name,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "gen_config": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "stream": stream_flag}
            })
            data[idx]["metadata"] = meta
            generated += 1

    elapsed = time.time() - start
    print(f"Generated {generated} outputs in {elapsed:.1f}s")
    write_jsonl(data, output_path)
    print(f"Wrote {len(data)} examples to {output_path}")


if __name__ == "__main__":
    main()