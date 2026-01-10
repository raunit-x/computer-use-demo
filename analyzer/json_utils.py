"""Utilities for extracting JSON from LLM responses."""

import json
from typing import Any


def extract_json_from_response(
    text: str,
    json_type: str = "object",
    default: Any = None,
) -> Any:
    """Extract and parse JSON from an LLM response that may contain markdown.
    
    Handles responses where JSON is wrapped in markdown code blocks (```json or ```).
    
    Args:
        text: The raw response text from an LLM.
        json_type: Type of JSON to extract - "object" for {...} or "array" for [...].
        default: Value to return if parsing fails.
        
    Returns:
        Parsed JSON data, or the default value if extraction/parsing fails.
    """
    text = text.strip()
    
    # Determine delimiters based on type
    if json_type == "array":
        start_delim, end_delim = "[", "]"
    else:
        start_delim, end_delim = "{", "}"
    
    # Try to find JSON directly in the text
    json_start = text.find(start_delim)
    json_end = text.rfind(end_delim) + 1
    
    # If not found or invalid, try extracting from code blocks
    if json_start == -1 or json_end <= json_start:
        text = _extract_from_code_block(text)
        json_start = text.find(start_delim)
        json_end = text.rfind(end_delim) + 1
    
    # Parse the JSON if found
    if json_start >= 0 and json_end > json_start:
        json_str = text[json_start:json_end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return default


def _extract_from_code_block(text: str) -> str:
    """Extract content from markdown code blocks.
    
    Handles both ```json and plain ``` code blocks.
    
    Args:
        text: Text that may contain markdown code blocks.
        
    Returns:
        The content inside the code block, or the original text if no block found.
    """
    # Try ```json first
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    
    # Try plain ```
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    
    return text

