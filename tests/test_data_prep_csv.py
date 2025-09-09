#!/usr/bin/env python3
"""
Tests for the CSV to JSONL data preparation functionality.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_fine_tune.data_prep import (
    extract_title_from_text,
    load_csv_rows,
    csv_row_to_jsonl_entry,
    process_csv_to_jsonl,
    CSV_DEFAULT_INSTRUCTION
)


def test_extract_title_from_text():
    """Test title extraction from text."""
    # Test with clear title
    text1 = "This is a Clear Title\n\nThis is the body of the text..."
    title1 = extract_title_from_text(text1)
    assert title1 == "This is a Clear Title"
    
    # Test with markdown header
    text2 = "# Markdown Header\n\nBody text here..."
    title2 = extract_title_from_text(text2)
    assert title2 == "Markdown Header"
    
    # Test with no clear title (use first sentence)
    text3 = "This is the first sentence. This is the second sentence."
    title3 = extract_title_from_text(text3)
    assert title3 == "This is the first sentence"
    
    # Test with very short text
    text4 = "Short"
    title4 = extract_title_from_text(text4)
    assert title4 == "Short"
    
    # Test with empty text
    title5 = extract_title_from_text("")
    assert title5 == ""


def test_csv_row_to_jsonl_entry_basic():
    """Test basic CSV row to JSONL conversion."""
    row = {
        "title": "Test Title",
        "full_text": "This is a test document with some content.",
        "output": "This is the expected output."
    }
    
    entry = csv_row_to_jsonl_entry(row)
    
    assert entry is not None
    assert entry["instruction"] == CSV_DEFAULT_INSTRUCTION
    assert "Title: Test Title" in entry["input"]
    assert "Text: This is a test document with some content." in entry["input"]
    assert entry["output"] == "This is the expected output."


def test_csv_row_to_jsonl_entry_no_title():
    """Test CSV row conversion when no title is present."""
    row = {
        "full_text": "Important Document Title\n\nThis is a test document with some content.",
    }
    
    entry = csv_row_to_jsonl_entry(row)
    
    assert entry is not None
    assert "Title: Important Document Title" in entry["input"]
    assert entry["output"] == ""  # No output field


def test_csv_row_to_jsonl_entry_section_ignored():
    """Test that section column is ignored."""
    row = {
        "title": "Test Title",
        "full_text": "This is a test document with some content.",
        "section": "This should be ignored",
        "output": "Expected output"
    }
    
    entry = csv_row_to_jsonl_entry(row)
    
    assert entry is not None
    assert "section" not in str(entry)
    assert "This should be ignored" not in str(entry)
    assert entry["output"] == "Expected output"


def test_csv_row_to_jsonl_entry_alternative_columns():
    """Test with alternative column names."""
    row = {
        "text": "This is content in the text column.",
        "label": "This is the label output."
    }
    
    entry = csv_row_to_jsonl_entry(row)
    
    assert entry is not None
    assert "Text: This is content in the text column." in entry["input"]
    assert entry["output"] == "This is the label output."


def test_csv_row_to_jsonl_entry_empty_text():
    """Test that rows with insufficient text are filtered out."""
    row = {
        "title": "Test",
        "full_text": "Short",  # Too short
    }
    
    entry = csv_row_to_jsonl_entry(row)
    assert entry is None


def test_load_csv_rows():
    """Test CSV loading functionality."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'full_text', 'output'])
        writer.writerow(['Test Title 1', 'This is test content 1', 'Output 1'])
        writer.writerow(['Test Title 2', 'This is test content 2', 'Output 2'])
        csv_path = Path(f.name)
    
    try:
        rows = load_csv_rows(csv_path)
        assert len(rows) == 2
        assert rows[0]['title'] == 'Test Title 1'
        assert rows[0]['full_text'] == 'This is test content 1'
        assert rows[1]['title'] == 'Test Title 2'
    finally:
        csv_path.unlink()


def test_process_csv_to_jsonl_integration():
    """Test end-to-end CSV to JSONL processing."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'full_text', 'summary'])
        writer.writerow(['Water Management Solutions', 'This document discusses innovative approaches to water conservation and management in urban areas. The solutions include rainwater harvesting, greywater recycling, and smart irrigation systems.', 'Comprehensive water management strategies'])
        writer.writerow(['Renewable Energy Systems', 'Solar and wind energy systems are becoming increasingly important for sustainable development. This text explores various implementation strategies and their effectiveness.', 'Analysis of renewable energy adoption'])
        csv_path = Path(f.name)
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        output_path = Path(f.name)
    
    try:
        # Process the CSV
        count = process_csv_to_jsonl(csv_path, output_path)
        assert count == 2
        
        # Verify the output
        assert output_path.exists()
        
        with output_path.open('r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Parse first entry
        entry1 = json.loads(lines[0])
        assert entry1["instruction"] == CSV_DEFAULT_INSTRUCTION
        assert "Title: Water Management Solutions" in entry1["input"]
        assert "water conservation" in entry1["input"]
        assert entry1["output"] == "Comprehensive water management strategies"
        
        # Parse second entry
        entry2 = json.loads(lines[1])
        assert "Title: Renewable Energy Systems" in entry2["input"]
        assert "Solar and wind energy" in entry2["input"]
        assert entry2["output"] == "Analysis of renewable energy adoption"
        
    finally:
        csv_path.unlink()
        if output_path.exists():
            output_path.unlink()


def test_process_csv_to_jsonl_filter_empty():
    """Test filtering entries with empty output."""
    # Create a temporary CSV file with mixed output availability
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'full_text', 'summary'])
        writer.writerow(['Title 1', 'This is content 1 with sufficient length to pass validation', 'Has output'])
        writer.writerow(['Title 2', 'This is content 2 with sufficient length to pass validation', ''])  # Empty output
        csv_path = Path(f.name)
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        output_path = Path(f.name)
    
    try:
        # Process without filtering
        count1 = process_csv_to_jsonl(csv_path, output_path, filter_empty_output=False)
        assert count1 == 2
        
        # Process with filtering
        count2 = process_csv_to_jsonl(csv_path, output_path, filter_empty_output=True)
        assert count2 == 1
        
        # Verify only entry with output remains
        with output_path.open('r') as f:
            lines = f.readlines()
        
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["output"] == "Has output"
        
    finally:
        csv_path.unlink()
        if output_path.exists():
            output_path.unlink()


def test_custom_instruction():
    """Test using custom instruction."""
    row = {
        "title": "Test Title",
        "full_text": "This is test content with sufficient length.",
    }
    
    custom_instruction = "Custom instruction for testing purposes."
    entry = csv_row_to_jsonl_entry(row, instruction=custom_instruction)
    
    assert entry is not None
    assert entry["instruction"] == custom_instruction


def test_alternative_text_column():
    """Test using alternative text column name."""
    row = {
        "title": "Test Title",
        "content": "This is content in alternative column.",
    }
    
    entry = csv_row_to_jsonl_entry(row, text_column="content")
    
    assert entry is not None
    assert "This is content in alternative column." in entry["input"]


if __name__ == "__main__":
    # Run tests directly
    import pytest
    pytest.main([__file__, "-v"])