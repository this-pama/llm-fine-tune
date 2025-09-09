#!/usr/bin/env python3
"""
Utility functions for LLM fine-tuning data preparation and text processing.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200, sentence_aware: bool = True) -> List[str]:
    """
    Split text into overlapping chunks with proper overlap preservation.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        sentence_aware: If True, prefer sentence boundaries for splitting
        
    Returns:
        List of text chunks with proper overlap
        
    Examples:
        >>> chunks = chunk_text("A" * 1000 + "B" * 1000, max_chars=800, overlap=100)
        >>> len(chunks[0])
        800
        >>> chunks[1][:100] == chunks[0][-100:]  # Overlap preserved
        True
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    
    if sentence_aware:
        return chunk_text_sentence_aware(text, max_chars, overlap)
    else:
        return chunk_text_character_based(text, max_chars, overlap)


def chunk_text_sentence_aware(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks preferring sentence boundaries to avoid mid-sentence splits.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks with sentence-aware splitting
    """
    # Sentence boundary patterns
    sentence_endings = re.compile(r'[.!?]+\s+')
    
    chunks = []
    start = 0
    max_iterations = len(text) // max(1, max_chars // 4)  # Prevent infinite loops
    iteration = 0
    
    while start < len(text) and iteration < max_iterations:
        iteration += 1
        end = min(start + max_chars, len(text))
        
        if end >= len(text):
            # Last chunk - take everything remaining
            final_chunk = text[start:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break
        
        # Look for sentence boundary within the chunk
        chunk_text = text[start:end]
        
        # Find sentence boundaries in the chunk text, searching backwards
        boundaries = []
        for match in sentence_endings.finditer(chunk_text):
            boundaries.append(match.end())
        
        if boundaries:
            # Use the last sentence boundary that leaves at least 25% of max_chars
            min_chunk_size = max_chars // 4
            for boundary in reversed(boundaries):
                if boundary >= min_chunk_size:
                    end = start + boundary
                    break
        else:
            # No sentence boundary found, try paragraph break
            nl_pos = text.rfind('\n\n', start, end)
            if nl_pos > start + max_chars // 4:
                end = nl_pos + 2  # Include the newlines
            elif text.rfind('\n', start, end) > start + max_chars // 4:
                # Single newline fallback
                end = text.rfind('\n', start, end) + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end >= len(text):
            break
        
        # Calculate next start with overlap
        next_start = end - overlap if end < len(text) else end
        # Ensure we make progress
        if next_start <= start:
            next_start = start + max(1, max_chars // 2)  # Force progress
        start = next_start
    
    return chunks


def chunk_text_character_based(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks using character-based splitting (legacy behavior).
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks with proper overlap
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        
        # Try to break at newline if we're not at the end
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start + max_chars // 4:
                end = nl
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        
        if end >= len(text):
            break
            
        # FIXED: Proper overlap calculation
        # Previous buggy line: start = max(end - overlap, end)  # This was wrong!
        # Correct overlap: next chunk starts overlap chars before current end
        next_start = end - overlap if end < len(text) else end
        start = max(next_start, start + 1)  # Ensure progress to avoid infinite loop
    
    return chunks


def extract_text_from_obj(response: Any) -> Optional[str]:
    """
    Extract text from various response object shapes (Ollama, OpenAI, HuggingFace).
    
    Handles common response patterns:
    - String responses
    - {"choices": [{"message": {"content": "..."}}]} (OpenAI format)
    - {"response": "..."} (Ollama format) 
    - {"generated_text": "..."} (HuggingFace format)
    - {"text": "..."} or {"content": "..."}
    
    Args:
        response: Response object from LLM API
        
    Returns:
        Extracted text string or None if no text found
        
    Examples:
        >>> extract_text_from_obj("Simple string")
        'Simple string'
        >>> extract_text_from_obj({"response": "Ollama output"})
        'Ollama output'
        >>> extract_text_from_obj({"choices": [{"message": {"content": "OpenAI"}}]})
        'OpenAI'
    """
    if response is None:
        return None
    
    # Direct string
    if isinstance(response, str):
        return response.strip() if response.strip() else None
    
    # Dictionary responses
    if isinstance(response, dict):
        # OpenAI format: choices[0].message.content
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if isinstance(choice, dict):
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    return content.strip() if isinstance(content, str) and content.strip() else None
                elif "text" in choice:
                    text = choice["text"]
                    return text.strip() if isinstance(text, str) and text.strip() else None
        
        # Ollama format: response
        if "response" in response:
            resp = response["response"]
            return resp.strip() if isinstance(resp, str) and resp.strip() else None
        
        # HuggingFace format: generated_text
        if "generated_text" in response:
            text = response["generated_text"]
            return text.strip() if isinstance(text, str) and text.strip() else None
        
        # Generic text/content fields
        for key in ["text", "content", "output", "result"]:
            if key in response:
                value = response[key]
                return value.strip() if isinstance(value, str) and value.strip() else None
    
    # List responses (take first valid text)
    if isinstance(response, list) and response:
        for item in response:
            result = extract_text_from_obj(item)
            if result:
                return result
    
    return None


def build_prompt(example: Dict[str, Any], template: Optional[str] = None) -> str:
    """
    Build a formatted prompt from an example using instruction-input-response template.
    
    Args:
        example: Dictionary with 'instruction' and 'input' keys
        template: Optional custom template, defaults to standard format
        
    Returns:
        Formatted prompt string
    """
    if template is None:
        template = (
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n"
        )
    
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    
    return template.format(instruction=instruction, input=input_text)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items: List[Dict[str, Any]], path: Path) -> None:
    """Write list of dictionaries to JSONL file."""
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    """Clean text by normalizing whitespace and line breaks."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s+\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def get_metadata_keys() -> Dict[str, str]:
    """
    Get canonical metadata keys used across scripts.
    
    Returns:
        Dictionary mapping semantic keys to standardized field names
    """
    return {
        "source_type": "source_type",          # solution, experiment, action_plan, blog
        "source_filename": "source_filename",  # original file name
        "title": "title",                      # document title
        "chunk_index": "chunk_index",          # chunk number in document
        "chunk_chars": "chunk_chars",          # character count of chunk
        "generated_by": "generated_by",        # model/service that generated output
        "generated_at": "generated_at",        # timestamp of generation
        "review_status": "review_status",      # manual review status
        "reviewer": "reviewer"                 # who reviewed the output
    }


# HTML stripping utility (duplicated from data_prep.py for now)
try:
    from bs4 import BeautifulSoup

    def strip_html(s: str) -> str:
        """Strip HTML tags using BeautifulSoup."""
        return BeautifulSoup(s, "html.parser").get_text(separator="\n")

except ImportError:
    # Fallback regex-based HTML stripping
    _TAG_RE = re.compile(r"<[^>]+>")

    def strip_html(s: str) -> str:
        """Strip HTML tags using regex (fallback)."""
        return _TAG_RE.sub("", s)
