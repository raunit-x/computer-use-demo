"""Prompts module for workflow automation.

This module centralizes all prompt templates used across the application,
organized by the module that uses them.
"""

from .analyzer_prompts import (
    EVENT_DETECTION_PROMPT,
    UNDERSTANDING_UPDATE_PROMPT,
    WORKFLOW_SYNTHESIS_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    PARAMETER_DETECTION_PROMPT,
)
from .executor_prompts import (
    WORKFLOW_SYSTEM_PROMPT,
)

__all__ = [
    # Analyzer prompts
    "EVENT_DETECTION_PROMPT",
    "UNDERSTANDING_UPDATE_PROMPT",
    "WORKFLOW_SYNTHESIS_PROMPT",
    "EXTRACTION_SYSTEM_PROMPT",
    "PARAMETER_DETECTION_PROMPT",
    # Executor prompts
    "WORKFLOW_SYSTEM_PROMPT",
]

