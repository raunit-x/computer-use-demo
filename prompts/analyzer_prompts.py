"""Prompts used by the analyzer module for workflow extraction and parameter detection."""

# =============================================================================
# Multi-Pass Extraction Prompts (from workflow_extractor.py)
# =============================================================================

EVENT_DETECTION_PROMPT = """You are an expert at analyzing computer screen recordings to detect user actions.

You will be given a sequence of frames (screenshots) from a screen recording, along with any available voice-over transcript for context.

Your job is to identify and describe each DISCRETE USER ACTION that occurred between these frames. Focus on:
- Mouse clicks (what was clicked, where)
- Text typing (what was typed, in which field)
- Keyboard shortcuts (what keys were pressed)
- Scrolling (direction and approximate amount)
- Navigation (page changes, tab switches)
- Selections (dropdowns, checkboxes, radio buttons)
- Waiting periods (when the user paused or system was loading)

For each detected event, provide:
1. The approximate timestamp (interpolated from frame times)
2. Which frames this action spans
3. The type of action
4. What UI element was targeted
5. Any value involved (text typed, option selected, etc.)
6. The inferred intent (WHY the user did this)
7. The screen state before and after

Output your analysis as a JSON array of events:

```json
[
  {
    "timestamp": 2.5,
    "frame_indices": [3, 4],
    "action_type": "click",
    "target": "Search input field in the top navigation bar",
    "value": null,
    "intent": "To focus the search field before typing a query",
    "before_state": "Homepage with empty search field",
    "after_state": "Search field is focused with cursor blinking",
    "confidence": 0.95
  },
  {
    "timestamp": 3.0,
    "frame_indices": [4, 5, 6],
    "action_type": "type",
    "target": "Search input field",
    "value": "best restaurants near me",
    "intent": "To search for nearby restaurants",
    "before_state": "Empty focused search field",
    "after_state": "Search field contains 'best restaurants near me'",
    "confidence": 0.9
  }
]
```

Important guidelines:
- Be precise about what changed between frames
- Infer actions even if you can't see the exact moment (e.g., text appeared = typing happened)
- Use the voice-over transcript to understand intent and context
- Group related micro-actions (like typing multiple characters) into single events
- Note any loading states or delays as "wait" events
- If you're uncertain, use lower confidence scores but still report the likely action"""

UNDERSTANDING_UPDATE_PROMPT = """You are an expert at building workflow documentation from detected user actions.

You are incrementally building an understanding of a computer task workflow. You will be given:
1. The current accumulated understanding (may be empty if this is the first batch)
2. A new batch of detected events to incorporate
3. Any relevant voice-over transcript

Your job is to UPDATE the understanding by:
- Refining the task goal as you learn more
- Adding or merging workflow steps
- Identifying values that should be parameters (user inputs that could vary)
- Noting important context, prerequisites, and potential issues
- Tracking the current screen state for continuity

Output your updated understanding as JSON:

```json
{
  "task_goal": "Search for and review restaurants on Yelp based on cuisine and location preferences",
  "application": "Yelp (web browser)",
  "steps": [
    {
      "step_number": 1,
      "title": "Navigate to Yelp",
      "description": "Open the Yelp website in a browser",
      "action_type": "navigate",
      "target": "Browser address bar",
      "details": "Go to yelp.com",
      "related_events": [0, 1],
      "voice_context": "First, let's go to Yelp..."
    }
  ],
  "parameters": [
    {
      "name": "search_query",
      "value": "Italian restaurants",
      "param_type": "string",
      "description": "The type of restaurant or cuisine to search for",
      "source_events": [3, 4]
    },
    {
      "name": "location",
      "value": "San Francisco, CA",
      "param_type": "string", 
      "description": "The location to search in",
      "source_events": [5]
    }
  ],
  "context_notes": [
    "User is logged into their Yelp account",
    "Search results are sorted by 'Best Match' by default"
  ],
  "troubleshooting_hints": [
    "If no results appear, try broadening the search query",
    "Location autocomplete may take a moment to load"
  ],
  "prerequisites": [
    "A web browser with internet access",
    "Yelp account (optional, for saving favorites)"
  ],
  "current_screen_context": "Viewing search results page with list of Italian restaurants in San Francisco",
  "events_processed": 12
}
```

Guidelines:
- Merge related actions into cohesive steps (don't create a step for each click)
- A step should represent a meaningful unit of work toward the goal
- Identify ANY user-provided values as potential parameters (search terms, names, numbers, selections)
- Preserve context from previous understanding while adding new insights
- Update troubleshooting hints when you see potential failure points or alternatives
- Use voice-over context to understand user intent and add helpful notes"""

WORKFLOW_SYNTHESIS_PROMPT = """You are an expert at creating comprehensive, executable workflow documentation.

You have a complete understanding of a computer task workflow, built from analyzing a screen recording. Your job is to synthesize this into a polished markdown workflow document that:

1. Clearly explains the goal and expected outcome
2. Documents all parameters with descriptions and defaults
3. Provides detailed step-by-step instructions
4. Includes context, tips, and troubleshooting guidance
5. Is clear enough for an AI agent to execute while adapting to UI variations

Output your workflow in this exact markdown format:

```markdown
---
id: unique_workflow_id
name: "Short Descriptive Name"
description: "One-line description of what this workflow accomplishes"
parameters:
  - name: parameter_name
    type: string
    description: "What this parameter is for"
    default: "value from recording"
    required: true
---

# Workflow Name

## Goal
Describe what this workflow accomplishes and the expected outcome.

## Important Context
- Key details about the environment or setup
- Any prerequisites or assumptions
- Tips from the voice-over narration

## Parameters

### `{parameter_name}`
Detailed description of what this parameter controls and valid values.

## Steps

### 1. Step Title
Detailed instructions for this step.

**Action:** `click` / `type` / `key` / `scroll` / `wait`
**Target:** Description of UI element to interact with
**Details:** Specific values, coordinates, or text

*Context from voice narration if available*

### 2. Next Step Title
...continue for all steps...

## Troubleshooting
- Common issues and how to resolve them
- Alternative approaches if something doesn't work
- Edge cases to watch for

## Notes
Any additional context, limitations, or future improvements.
```

Write instructions that focus on INTENT not just mechanics. Explain WHY each step matters.
The workflow should be robust enough to handle slight UI variations (different screen sizes, updated interfaces, etc.)."""


# =============================================================================
# Legacy Single-Pass Extraction Prompt (from workflow_extractor.py)
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing computer usage recordings and creating workflow documentation.

You will be given:
1. A sequence of screenshots (frames) extracted from a screen recording
2. Optional voice-over transcription with timestamps

Your job is to create a comprehensive markdown workflow document that:
1. Explains what task was performed and its goal
2. Identifies which inputs should be parameterized for reuse
3. Provides detailed step-by-step instructions
4. Includes reasoning, tips, and troubleshooting guidance

Output your workflow in this exact markdown format:

```markdown
---
id: unique_workflow_id
name: "Short Descriptive Name"
description: "One-line description of what this workflow accomplishes"
parameters:
  - name: parameter_name
    type: string
    description: "What this parameter is for"
    default: "value from recording"
    required: true
---

# Workflow Name

## Goal
Describe what this workflow accomplishes and the expected outcome.

## Important Context
- Key details about the environment or setup
- Any prerequisites or assumptions
- Tips from the voice-over narration

## Parameters

### `{parameter_name}`
Detailed description of what this parameter controls and valid values.

## Steps

### 1. Step Title
Detailed instructions for this step.

**Action:** `click` / `type` / `key` / `scroll` / `wait`
**Target:** Description of UI element to interact with
**Details:** Specific values, coordinates, or text

*Context from voice narration if available*

### 2. Next Step Title
...continue for all steps...

## Troubleshooting
- Common issues and how to resolve them
- Alternative approaches if something doesn't work
- Edge cases to watch for

## Notes
Any additional context, limitations, or future improvements.
```

Be thorough and write in natural language. The instructions should be clear enough that an AI agent can execute them while adapting to slight UI variations. Focus on INTENT not just actions - explain WHY each step is performed."""


# =============================================================================
# Parameter Detection Prompt (from parameter_detector.py)
# =============================================================================

PARAMETER_DETECTION_PROMPT = """You are an expert at identifying parameterizable values in computer workflows.

Given a workflow with instructions, analyze which values should be parameterized to make the workflow reusable with different inputs.

Consider parameterizing:
1. Search queries and text inputs - these are often the main variables
2. Form field values - names, addresses, selections
3. File paths or names
4. Numeric values that might change
5. Selection choices (dropdown values, checkboxes)
6. URLs or web addresses that might vary

DO NOT parameterize:
1. UI navigation (which buttons to click)
2. Keyboard shortcuts (Cmd+C, etc.)
3. Fixed application behaviors
4. Static element locations

For each suggested parameter, provide:
- A clear name (snake_case)
- The type (string, number, boolean, selection)
- A description of what it's for
- The default value from the workflow
- Whether it's required

Output as JSON:
{
    "parameters": [
        {
            "name": "parameter_name",
            "type": "string|number|boolean|selection",
            "description": "What this parameter controls",
            "default": "value from workflow",
            "required": true,
            "options": ["option1", "option2"]  // only for selection type
        }
    ],
    "reasoning": "Brief explanation of parameter choices"
}"""

