"""Workflow extraction from video recordings using Claude or OpenAI."""

import base64
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anthropic import Anthropic
from PIL import Image
from tqdm import tqdm

from .schema import (
    DetectedEvent,
    RunningUnderstanding,
    Workflow,
)
from .json_utils import extract_json_from_response
from prompts.analyzer_prompts import (
    EVENT_DETECTION_PROMPT,
    UNDERSTANDING_UPDATE_PROMPT,
    WORKFLOW_SYNTHESIS_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from recorder.video_processor import ProcessedSession
    from utils.logger import WorkflowLogger
    from utils.tracking import CostTracker


class WorkflowExtractor:
    """Extracts structured workflows from video recordings using Claude or OpenAI."""
    
    # Maximum image dimension for API requests (to avoid request size limits)
    MAX_IMAGE_DIMENSION = 1280
    # JPEG quality for compressed images
    IMAGE_QUALITY = 85
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        use_openai: bool = False,
        max_image_dimension: int | None = None,
        logger: "WorkflowLogger | None" = None,
        cost_tracker: "CostTracker | None" = None,
    ):
        """Initialize the extractor.
        
        Args:
            api_key: API key. If not provided, uses ANTHROPIC_API_KEY or OPENAI_API_KEY env var.
            model: Model to use for extraction.
            use_openai: If True, use OpenAI API instead of Anthropic.
            max_image_dimension: Maximum dimension for resized images (default 1280).
            logger: Optional WorkflowLogger for structured output.
            cost_tracker: Optional CostTracker for cost accumulation.
        """
        self.use_openai = use_openai
        self.model = model
        self.max_image_dimension = max_image_dimension or self.MAX_IMAGE_DIMENSION
        self.logger = logger
        self.cost_tracker = cost_tracker
        
        if use_openai:
            from openai import OpenAI
            self.client: Any = OpenAI(api_key=api_key) if api_key else OpenAI()
        else:
            self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log a message using the logger or print."""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "step":
                self.logger.step(message)
            elif level == "success":
                self.logger.success(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "header":
                self.logger.header(message)
        else:
            print(message)
    
    def _track_usage(self, input_tokens: int, output_tokens: int, phase: str) -> None:
        """Track API usage if cost_tracker is available."""
        if self.cost_tracker:
            self.cost_tracker.add_usage(input_tokens, output_tokens, phase=phase)
        if self.logger:
            self.logger.api(input_tokens, output_tokens)
    
    def _resize_and_encode_image(self, image_path: Path) -> tuple[str, str]:
        """Resize image if needed and return base64-encoded data.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Tuple of (base64_data, media_type).
        """
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for JPEG encoding)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Check if resizing is needed
            width, height = img.size
            max_dim = max(width, height)
            
            if max_dim > self.max_image_dimension:
                # Calculate new dimensions maintaining aspect ratio
                scale = self.max_image_dimension / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                # Use LANCZOS resampling (Resampling.LANCZOS in newer Pillow versions)
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS  # type: ignore[attr-defined]
                img = img.resize((new_width, new_height), resample)
            
            # Encode as JPEG for smaller size
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=self.IMAGE_QUALITY, optimize=True)
            buffer.seek(0)
            
            image_data = base64.b64encode(buffer.read()).decode()
            return image_data, "image/jpeg"
    
    def extract_from_video(
        self,
        video_path: Path,
        output_dir: Path | None = None,
        max_frames: int | None = None,  # Deprecated, kept for compatibility
        openai_api_key: str | None = None,
        chunk_size: int = 10,
        verbose: bool = True,
    ) -> Workflow:
        """Extract a workflow from a video file.
        
        This is the main entry point for video-based workflow extraction.
        It processes the video to extract frames and audio, then uses
        multi-pass analysis to build a comprehensive workflow.
        
        Args:
            video_path: Path to the video file (.mov, .mp4, etc.).
            output_dir: Directory to store processed data. If None, uses temp dir.
            max_frames: Deprecated. The multi-pass approach processes all frames.
            openai_api_key: OpenAI API key for audio transcription.
            chunk_size: Number of frames to analyze per chunk in Pass 1.
            verbose: Whether to print progress information with tqdm bars.
            
        Returns:
            Extracted Workflow object.
        """
        # Import here to avoid circular imports
        from recorder.video_processor import VideoProcessor
        
        # Process the video
        processor = VideoProcessor(
            fps=2.0,  # Extract 2 frames per second
            openai_api_key=openai_api_key,
        )
        
        session = processor.process(
            video_path=video_path,
            output_dir=output_dir,
            extract_audio=True,
            transcribe=True,
        )
        
        return self.extract_from_processed_session(
            session,
            chunk_size=chunk_size,
            verbose=verbose,
        )
    
    def extract_from_processed_session(
        self,
        session: "ProcessedSession",
        max_frames: int | None = None,  # Deprecated, kept for compatibility
        chunk_size: int = 10,
        verbose: bool = True,
    ) -> Workflow:
        """Extract a workflow from a processed video session using multi-pass analysis.
        
        This method uses a three-pass approach:
        1. Event Detection: Analyze frames in chunks to detect discrete user actions
        2. Understanding Building: Incrementally build workflow understanding from events
        3. Workflow Synthesis: Generate polished markdown workflow document
        
        Args:
            session: ProcessedSession from VideoProcessor.
            max_frames: Deprecated. The multi-pass approach processes all frames.
            chunk_size: Number of frames to analyze per chunk in Pass 1.
            verbose: Whether to print progress information with tqdm bars.
            
        Returns:
            Extracted Workflow object.
        """
        frames = session.frames
        
        if verbose:
            self._log("MULTI-PASS WORKFLOW EXTRACTION", level="header")
            self._log(f"Session ID: {session.session_id}")
            self._log(f"Duration: {session.duration:.1f}s")
            self._log(f"Total frames: {len(frames)}")
            if session.transcript:
                self._log(f"Transcript: {len(session.transcript.segments)} segments")
        
        if not frames:
            # Return empty workflow if no frames
            return Workflow(
                id="empty_workflow",
                name="Empty Workflow",
                description="No frames were provided for analysis",
                parameters=[],
                instructions="No content available.",
                source_session_id=session.session_id,
            )
        
        # Pass 1: Detect events from all frames
        events = self._detect_events_pass(
            frames=frames,
            transcript=session.transcript,
            chunk_size=chunk_size,
            overlap=2,
            verbose=verbose,
        )
        
        # Pass 2: Build running understanding from events
        understanding = self._build_understanding_pass(
            events=events,
            transcript=session.transcript,
            batch_size=15,
            verbose=verbose,
        )
        
        # Pass 3: Generate final workflow
        workflow = self._generate_workflow_pass(
            understanding=understanding,
            verbose=verbose,
        )
        
        if verbose:
            self._log("EXTRACTION COMPLETE", level="header")
            self._log(f"Generated workflow: {workflow.name}", level="success")
            self._log(f"  - {len(workflow.parameters)} parameters")
            self._log(f"  - {len(workflow.instructions)} chars of instructions")
        
        workflow.source_session_id = session.session_id
        return workflow
    
    def extract_from_processed_session_legacy(
        self,
        session: "ProcessedSession",
        max_frames: int = 30,
    ) -> Workflow:
        """Legacy single-pass extraction (kept for backward compatibility).
        
        This method uses the original single-pass approach where frames are
        sampled and sent in a single API call. Use extract_from_processed_session
        for better results with the multi-pass approach.
        
        Args:
            session: ProcessedSession from VideoProcessor.
            max_frames: Maximum number of frames to analyze.
            
        Returns:
            Extracted Workflow object.
        """
        # Sample frames if too many
        frames = session.frames
        if len(frames) > max_frames:
            step = len(frames) // max_frames
            frames = frames[::step][:max_frames]
        
        if self.use_openai:
            # Build OpenAI-formatted message content
            content = self._build_openai_message_from_session(
                frames=frames,
                transcript=session.transcript,
                duration=session.duration,
            )
            response_text = self._call_openai(content)
        else:
            # Build Anthropic-formatted message content
            content = self._build_extraction_message_from_session(
                frames=frames,
                transcript=session.transcript,
                duration=session.duration,
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            response_text = response.content[0].text
        
        # Parse the markdown response
        markdown_content = self._extract_markdown(response_text)
        
        # Create workflow from markdown
        workflow = Workflow.from_markdown(markdown_content)
        workflow.source_session_id = session.session_id
        
        return workflow
    
    def extract_from_session(
        self,
        session_dir: Path,
        max_screenshots: int = 20,
        chunk_size: int = 10,
        verbose: bool = True,
    ) -> Workflow:
        """Extract a workflow from a legacy recorded session directory.
        
        This method is kept for backward compatibility with the old recording format.
        For new processed sessions, it uses the multi-pass extraction approach.
        
        Args:
            session_dir: Path to the session directory with session.json.
            max_screenshots: Maximum number of screenshots for legacy format.
            chunk_size: Number of frames per chunk for multi-pass analysis.
            verbose: Whether to print progress information with tqdm bars.
            
        Returns:
            Extracted Workflow object.
        """
        # Check if this is a new processed session or legacy format
        processed_json = session_dir / "processed_session.json"
        session_json = session_dir / "session.json"
        
        if processed_json.exists():
            # New format - load ProcessedSession and use multi-pass extraction
            from recorder.video_processor import ProcessedSession
            session = ProcessedSession.load(session_dir)
            return self.extract_from_processed_session(
                session,
                chunk_size=chunk_size,
                verbose=verbose,
            )
        
        if not session_json.exists():
            raise FileNotFoundError(f"No session data found in {session_dir}")
        
        # Legacy format
        with open(session_json) as f:
            session_data = json.load(f)
        
        # Get screenshot paths
        screenshots_dir = session_dir / "screenshots"
        screenshot_paths = sorted(screenshots_dir.glob("*.png")) if screenshots_dir.exists() else []
        
        # Also check for frames directory (new format)
        frames_dir = session_dir / "frames"
        if frames_dir.exists():
            screenshot_paths = sorted(frames_dir.glob("*.png"))
        
        # Sample screenshots if too many
        if len(screenshot_paths) > max_screenshots:
            step = len(screenshot_paths) // max_screenshots
            screenshot_paths = screenshot_paths[::step][:max_screenshots]
        
        if self.use_openai:
            # Build OpenAI-formatted message content
            content = self._build_openai_extraction_message(
                session_data=session_data,
                screenshot_paths=screenshot_paths,
            )
            response_text = self._call_openai(content)
        else:
            # Build the message content
            content = self._build_extraction_message(
                session_data=session_data,
                screenshot_paths=screenshot_paths,
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            response_text = response.content[0].text
        
        # Parse the markdown response
        markdown_content = self._extract_markdown(response_text)
        
        # Create workflow from markdown
        workflow = Workflow.from_markdown(markdown_content)
        workflow.source_session_id = session_data.get("session_id")
        
        return workflow
    
    def _build_extraction_message_from_session(
        self,
        frames: list,  # List of FrameInfo
        transcript,  # Transcript or None
        duration: float,
    ) -> list[dict]:
        """Build message content from a processed session."""
        content = []
        
        # Add text description
        text_parts = ["# Video Recording Analysis\n\n"]
        text_parts.append(f"Duration: {duration:.1f} seconds\n")
        text_parts.append(f"Frames extracted: {len(frames)}\n\n")
        
        # Add transcript if available
        if transcript and transcript.text:
            text_parts.append("## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript.text}\n")
            
            if transcript.segments:
                text_parts.append("\n### Segments with timestamps:\n")
                for seg in transcript.segments:
                    text_parts.append(f"- [{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}\n")
        else:
            text_parts.append("## Voice-Over\n\nNo audio narration was detected in this recording.\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add frames as images
        content.append({
            "type": "text",
            "text": f"\n## Screenshots ({len(frames)} frames)\n\nBelow are frames from the recording in chronological order:\n"
        })
        
        for i, frame in enumerate(frames):
            # Resize and compress image to avoid API size limits
            image_data, media_type = self._resize_and_encode_image(frame.path)
            
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                }
            })
            content.append({
                "type": "text",
                "text": f"Frame {i + 1} at {frame.timestamp:.1f}s"
            })
        
        # Add extraction instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease analyze this recording and create a comprehensive markdown workflow document. Include the YAML frontmatter with parameters, then detailed instructions with reasoning."
        })
        
        return content
    
    def _build_openai_message_from_session(
        self,
        frames: list,  # List of FrameInfo
        transcript,  # Transcript or None
        duration: float,
    ) -> list[dict]:
        """Build message content in OpenAI's vision API format."""
        content = []
        
        # Add text description
        text_parts = ["# Video Recording Analysis\n\n"]
        text_parts.append(f"Duration: {duration:.1f} seconds\n")
        text_parts.append(f"Frames extracted: {len(frames)}\n\n")
        
        # Add transcript if available
        if transcript and transcript.text:
            text_parts.append("## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript.text}\n")
            
            if transcript.segments:
                text_parts.append("\n### Segments with timestamps:\n")
                for seg in transcript.segments:
                    text_parts.append(f"- [{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}\n")
        else:
            text_parts.append("## Voice-Over\n\nNo audio narration was detected in this recording.\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add frames as images (OpenAI format)
        content.append({
            "type": "text",
            "text": f"\n## Screenshots ({len(frames)} frames)\n\nBelow are frames from the recording in chronological order:\n"
        })
        
        for i, frame in enumerate(frames):
            # Resize and compress image to avoid API size limits
            image_data, media_type = self._resize_and_encode_image(frame.path)
            
            # OpenAI uses image_url with data URL format
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_data}"
                }
            })
            content.append({
                "type": "text",
                "text": f"Frame {i + 1} at {frame.timestamp:.1f}s"
            })
        
        # Add extraction instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease analyze this recording and create a comprehensive markdown workflow document. Include the YAML frontmatter with parameters, then detailed instructions with reasoning."
        })
        
        return content
    
    def _call_openai(self, content: list[dict]) -> str:
        """Call OpenAI API with vision content and return response text."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=8192,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        return response.choices[0].message.content
    
    def _build_extraction_message(
        self,
        session_data: dict,
        screenshot_paths: list[Path],
    ) -> list[dict]:
        """Build the message content for extraction (legacy format)."""
        content = []
        
        # Add text description
        text_parts = ["# Recording Session Analysis\n\n"]
        text_parts.append(f"Session ID: {session_data.get('session_id', 'unknown')}\n")
        text_parts.append(f"Duration: {session_data.get('duration', 0):.1f} seconds\n\n")
        
        # Add events summary if present
        events = session_data.get("events", [])
        if events:
            text_parts.append("## Input Events\n\n")
            for event in events[:100]:
                timestamp = event.get("timestamp", 0)
                if "start_time" in session_data:
                    timestamp = timestamp - session_data["start_time"]
                text_parts.append(f"- [{timestamp:.2f}s] {event.get('event_type', 'unknown')}: {event.get('data', {})}\n")
        
        # Add transcript if available
        if session_data.get("transcript"):
            transcript = session_data["transcript"]
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript.get('text', '')}\n")
            
            if transcript.get("segments"):
                text_parts.append("\n### Segments with timestamps:\n")
                for seg in transcript["segments"]:
                    text_parts.append(f"- [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add screenshots
        if screenshot_paths:
            content.append({
                "type": "text",
                "text": f"\n## Screenshots ({len(screenshot_paths)} images)\n\nBelow are screenshots from the recording in chronological order:\n"
            })
            
            for i, path in enumerate(screenshot_paths):
                # Resize and compress image to avoid API size limits
                image_data, media_type = self._resize_and_encode_image(path)
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"Screenshot {i + 1} of {len(screenshot_paths)}"
                })
        
        # Add extraction instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease analyze this recording and create a comprehensive markdown workflow document. Include the YAML frontmatter with parameters, then detailed instructions with reasoning."
        })
        
        return content
    
    def _build_openai_extraction_message(
        self,
        session_data: dict,
        screenshot_paths: list[Path],
    ) -> list[dict]:
        """Build the message content for extraction in OpenAI format (legacy format)."""
        content = []
        
        # Add text description
        text_parts = ["# Recording Session Analysis\n\n"]
        text_parts.append(f"Session ID: {session_data.get('session_id', 'unknown')}\n")
        text_parts.append(f"Duration: {session_data.get('duration', 0):.1f} seconds\n\n")
        
        # Add events summary if present
        events = session_data.get("events", [])
        if events:
            text_parts.append("## Input Events\n\n")
            for event in events[:100]:
                timestamp = event.get("timestamp", 0)
                if "start_time" in session_data:
                    timestamp = timestamp - session_data["start_time"]
                text_parts.append(f"- [{timestamp:.2f}s] {event.get('event_type', 'unknown')}: {event.get('data', {})}\n")
        
        # Add transcript if available
        if session_data.get("transcript"):
            transcript = session_data["transcript"]
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript.get('text', '')}\n")
            
            if transcript.get("segments"):
                text_parts.append("\n### Segments with timestamps:\n")
                for seg in transcript["segments"]:
                    text_parts.append(f"- [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add screenshots (OpenAI format)
        if screenshot_paths:
            content.append({
                "type": "text",
                "text": f"\n## Screenshots ({len(screenshot_paths)} images)\n\nBelow are screenshots from the recording in chronological order:\n"
            })
            
            for i, path in enumerate(screenshot_paths):
                # Resize and compress image to avoid API size limits
                image_data, media_type = self._resize_and_encode_image(path)
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"Screenshot {i + 1} of {len(screenshot_paths)}"
                })
        
        # Add extraction instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease analyze this recording and create a comprehensive markdown workflow document. Include the YAML frontmatter with parameters, then detailed instructions with reasoning."
        })
        
        return content
    
    def extract_from_data(
        self,
        screenshots: list[tuple[Path, dict]],
        events: list[dict],
        transcript_text: str | None = None,
        transcript_segments: list[dict] | None = None,
    ) -> Workflow:
        """Extract a workflow from raw data (legacy method).
        
        Args:
            screenshots: List of (screenshot_path, metadata) tuples.
            events: List of input events.
            transcript_text: Full transcript text.
            transcript_segments: Transcript segments with timestamps.
            
        Returns:
            Extracted Workflow object.
        """
        if self.use_openai:
            content = self._build_openai_message_from_data(
                screenshots=screenshots,
                events=events,
                transcript_text=transcript_text,
                transcript_segments=transcript_segments,
            )
            response_text = self._call_openai(content)
        else:
            content = self._build_extraction_message_from_data(
                screenshots=screenshots,
                events=events,
                transcript_text=transcript_text,
                transcript_segments=transcript_segments,
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            response_text = response.content[0].text
        
        markdown_content = self._extract_markdown(response_text)
        return Workflow.from_markdown(markdown_content)
    
    def _build_extraction_message_from_data(
        self,
        screenshots: list[tuple[Path, dict]],
        events: list[dict],
        transcript_text: str | None,
        transcript_segments: list[dict] | None,
    ) -> list[dict]:
        """Build message content from raw data."""
        content = []
        
        text_parts = ["# Recording Analysis\n\n"]
        
        # Events
        if events:
            text_parts.append("## Input Events\n\n")
            for event in events[:100]:
                text_parts.append(f"- [{event.get('timestamp', 0):.2f}s] {event.get('event_type', 'unknown')}: {event.get('data', {})}\n")
        
        # Transcript
        if transcript_text:
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript_text}\n")
            
            if transcript_segments:
                text_parts.append("\n### Segments:\n")
                for seg in transcript_segments:
                    text_parts.append(f"- [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Screenshots
        if screenshots:
            content.append({
                "type": "text",
                "text": f"\n## Screenshots ({len(screenshots)} images)\n\n"
            })
            
            for i, (path, metadata) in enumerate(screenshots):
                # Resize and compress image to avoid API size limits
                image_data, media_type = self._resize_and_encode_image(path)
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"Screenshot {i + 1}: trigger={metadata.get('trigger', 'unknown')}"
                })
        
        content.append({
            "type": "text",
            "text": "\n\nCreate a comprehensive markdown workflow document with YAML frontmatter and detailed instructions."
        })
        
        return content
    
    def _build_openai_message_from_data(
        self,
        screenshots: list[tuple[Path, dict]],
        events: list[dict],
        transcript_text: str | None,
        transcript_segments: list[dict] | None,
    ) -> list[dict]:
        """Build message content from raw data in OpenAI format."""
        content = []
        
        text_parts = ["# Recording Analysis\n\n"]
        
        # Events
        if events:
            text_parts.append("## Input Events\n\n")
            for event in events[:100]:
                text_parts.append(f"- [{event.get('timestamp', 0):.2f}s] {event.get('event_type', 'unknown')}: {event.get('data', {})}\n")
        
        # Transcript
        if transcript_text:
            text_parts.append("\n## Voice-Over Transcript\n\n")
            text_parts.append(f"{transcript_text}\n")
            
            if transcript_segments:
                text_parts.append("\n### Segments:\n")
                for seg in transcript_segments:
                    text_parts.append(f"- [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Screenshots (OpenAI format)
        if screenshots:
            content.append({
                "type": "text",
                "text": f"\n## Screenshots ({len(screenshots)} images)\n\n"
            })
            
            for i, (path, metadata) in enumerate(screenshots):
                # Resize and compress image to avoid API size limits
                image_data, media_type = self._resize_and_encode_image(path)
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"Screenshot {i + 1}: trigger={metadata.get('trigger', 'unknown')}"
                })
        
        content.append({
            "type": "text",
            "text": "\n\nCreate a comprehensive markdown workflow document with YAML frontmatter and detailed instructions."
        })
        
        return content
    
    def _extract_markdown(self, response_text: str) -> str:
        """Extract markdown content from Claude's response."""
        text = response_text.strip()
        
        # If response is wrapped in code block, extract it
        if text.startswith("```markdown"):
            start = text.find("```markdown") + 11
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
        elif text.startswith("```"):
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1:
                start = newline + 1
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
        
        # Ensure we have frontmatter
        if not text.startswith("---"):
            # Try to add minimal frontmatter
            text = f"---\nid: extracted_workflow\nname: Extracted Workflow\n---\n\n{text}"
        
        return text
    
    # =========================================================================
    # Multi-Pass Extraction Methods
    # =========================================================================
    
    def _detect_events_pass(
        self,
        frames: list,  # List of FrameInfo
        transcript,  # Transcript or None
        chunk_size: int = 10,
        overlap: int = 2,
        verbose: bool = True,
    ) -> list[DetectedEvent]:
        """Pass 1: Detect events from frames in overlapping chunks.
        
        Processes all frames in chunks, with overlap to catch actions that
        span chunk boundaries. Each chunk is analyzed independently to
        identify discrete user actions.
        
        Args:
            frames: List of FrameInfo objects from the video.
            transcript: Optional Transcript with voice-over text.
            chunk_size: Number of frames per chunk.
            overlap: Number of frames to overlap between chunks.
            verbose: Whether to print progress information.
            
        Returns:
            List of DetectedEvent objects in chronological order.
        """
        all_events: list[DetectedEvent] = []
        total_frames = len(frames)
        
        if total_frames == 0:
            return all_events
        
        # Create chunks with overlap
        chunks = self._create_overlapping_chunks(frames, chunk_size, overlap)
        total_chunks = len(chunks)
        
        if verbose:
            self._log("PASS 1: Event Detection", level="header")
            self._log(f"Total frames: {total_frames}")
            self._log(f"Chunk size: {chunk_size} frames (overlap: {overlap})")
            self._log(f"Total chunks to process: {total_chunks}")
        
        # Use tqdm for progress bar
        chunk_iterator = tqdm(
            enumerate(chunks),
            total=total_chunks,
            desc="Detecting events",
            unit="chunk",
            disable=not verbose,
        )
        
        for chunk_idx, (chunk_frames, start_idx) in chunk_iterator:
            # Update progress bar description
            chunk_iterator.set_postfix({
                "frames": f"{start_idx + 1}-{start_idx + len(chunk_frames)}",
                "events": len(all_events),
            })
            
            # Get relevant transcript segments for this chunk
            chunk_start_time = chunk_frames[0].timestamp
            chunk_end_time = chunk_frames[-1].timestamp
            relevant_transcript = self._get_transcript_for_timerange(
                transcript, chunk_start_time, chunk_end_time
            )
            
            # Detect events in this chunk
            chunk_events = self._detect_events_in_chunk(
                frames=chunk_frames,
                start_frame_idx=start_idx,
                transcript_text=relevant_transcript,
            )
            
            # Merge events, handling overlap with previous chunk
            all_events = self._merge_events(all_events, chunk_events, overlap > 0)
        
        if verbose:
            self._log(f"Pass 1 complete: Detected {len(all_events)} events", level="success")
        
        return all_events
    
    def _create_overlapping_chunks(
        self,
        frames: list,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[list, int]]:
        """Create overlapping chunks of frames.
        
        Returns list of (chunk_frames, start_index) tuples.
        """
        chunks = []
        step = max(1, chunk_size - overlap)
        
        for i in range(0, len(frames), step):
            chunk = frames[i:i + chunk_size]
            if len(chunk) > 1:  # Need at least 2 frames to detect changes
                chunks.append((chunk, i))
        
        return chunks
    
    def _get_transcript_for_timerange(
        self,
        transcript,
        start_time: float,
        end_time: float,
    ) -> str:
        """Extract transcript text relevant to a time range."""
        if not transcript or not transcript.segments:
            return ""
        
        relevant_parts = []
        for segment in transcript.segments:
            # Check if segment overlaps with our time range
            if segment.end >= start_time and segment.start <= end_time:
                relevant_parts.append(f"[{segment.start:.1f}s]: {segment.text}")
        
        return "\n".join(relevant_parts)
    
    def _detect_events_in_chunk(
        self,
        frames: list,
        start_frame_idx: int,
        transcript_text: str,
    ) -> list[DetectedEvent]:
        """Detect events in a single chunk of frames."""
        # Build message content
        content = self._build_event_detection_message(
            frames=frames,
            start_frame_idx=start_frame_idx,
            transcript_text=transcript_text,
        )
        
        # Call the appropriate API
        if self.use_openai:
            response_text = self._call_openai_with_prompt(
                content, EVENT_DETECTION_PROMPT, phase="pass1_events"
            )
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=EVENT_DETECTION_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            response_text = response.content[0].text
            # Track usage
            self._track_usage(
                response.usage.input_tokens,
                response.usage.output_tokens,
                phase="pass1_events",
            )
        
        # Parse JSON response
        return self._parse_events_response(response_text)
    
    def _build_event_detection_message(
        self,
        frames: list,
        start_frame_idx: int,
        transcript_text: str,
    ) -> list[dict]:
        """Build message content for event detection."""
        content = []
        
        # Add context
        text_parts = [f"# Frame Chunk Analysis\n\n"]
        text_parts.append(f"Analyzing frames {start_frame_idx + 1} to {start_frame_idx + len(frames)}\n")
        text_parts.append(f"Time range: {frames[0].timestamp:.1f}s to {frames[-1].timestamp:.1f}s\n\n")
        
        if transcript_text:
            text_parts.append("## Relevant Voice-Over:\n")
            text_parts.append(f"{transcript_text}\n\n")
        
        content.append({"type": "text", "text": "".join(text_parts)})
        
        # Add frames
        content.append({
            "type": "text",
            "text": f"## Frames ({len(frames)} images)\n\n"
        })
        
        for i, frame in enumerate(frames):
            # Resize and compress image to avoid API size limits
            image_data, media_type = self._resize_and_encode_image(frame.path)
            
            if self.use_openai:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_data}"}
                })
            else:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    }
                })
            
            content.append({
                "type": "text",
                "text": f"Frame {start_frame_idx + i + 1} at {frame.timestamp:.1f}s"
            })
        
        content.append({
            "type": "text",
            "text": "\n\nAnalyze the changes between these frames and detect all user actions. Output as JSON array."
        })
        
        return content
    
    def _call_openai_with_prompt(
        self,
        content: list[dict],
        system_prompt: str,
        phase: str = "openai",
    ) -> str:
        """Call OpenAI API with custom system prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=16384,  # Increased from 4096 to handle growing understanding
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        # Track usage if available
        if response.usage:
            self._track_usage(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                phase=phase,
            )
        return response.choices[0].message.content
    
    def _parse_events_response(self, response_text: str) -> list[DetectedEvent]:
        """Parse JSON events from model response."""
        events_data = extract_json_from_response(response_text, json_type="array", default=[])
        return [DetectedEvent.from_dict(event_dict) for event_dict in events_data]
    
    def _merge_events(
        self,
        existing_events: list[DetectedEvent],
        new_events: list[DetectedEvent],
        has_overlap: bool,
    ) -> list[DetectedEvent]:
        """Merge new events with existing events, handling overlap."""
        if not existing_events:
            return new_events
        
        if not new_events:
            return existing_events
        
        if not has_overlap:
            return existing_events + new_events
        
        # Check for duplicate events in the overlap region
        # Use timestamp and action_type to detect duplicates
        last_existing_time = existing_events[-1].timestamp
        
        merged = list(existing_events)
        for event in new_events:
            # Skip if this looks like a duplicate from overlap
            is_duplicate = False
            if event.timestamp <= last_existing_time:
                for existing in existing_events[-3:]:  # Check last few events
                    if (abs(event.timestamp - existing.timestamp) < 0.5 and
                        event.action_type == existing.action_type and
                        event.target == existing.target):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                merged.append(event)
        
        return merged
    
    def _build_understanding_pass(
        self,
        events: list[DetectedEvent],
        transcript,  # Transcript or None
        batch_size: int = 15,
        verbose: bool = True,
    ) -> RunningUnderstanding:
        """Pass 2: Build running understanding from detected events.
        
        Processes events in batches, incrementally building and refining
        the workflow understanding. Each batch updates the accumulated state.
        
        Args:
            events: List of DetectedEvent from Pass 1.
            transcript: Optional Transcript for additional context.
            batch_size: Number of events per batch.
            verbose: Whether to print progress information.
            
        Returns:
            Complete RunningUnderstanding of the workflow.
        """
        understanding = RunningUnderstanding.empty()
        
        if not events:
            return understanding
        
        # Process events in batches
        total_events = len(events)
        num_batches = (total_events + batch_size - 1) // batch_size
        
        if verbose:
            self._log("PASS 2: Building Understanding", level="header")
            self._log(f"Total events to process: {total_events}")
            self._log(f"Batch size: {batch_size} events")
            self._log(f"Total batches: {num_batches}")
        
        # Use tqdm for progress bar
        batch_iterator = tqdm(
            range(num_batches),
            desc="Building understanding",
            unit="batch",
            disable=not verbose,
        )
        
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_events)
            batch_events = events[start_idx:end_idx]
            
            # Update progress bar description
            batch_iterator.set_postfix({
                "events": f"{start_idx + 1}-{end_idx}",
                "steps": len(understanding.steps),
            })
            
            # Get transcript context for this batch's time range
            if batch_events:
                batch_start_time = batch_events[0].timestamp
                batch_end_time = batch_events[-1].timestamp
                transcript_context = self._get_transcript_for_timerange(
                    transcript, batch_start_time, batch_end_time
                )
            else:
                transcript_context = ""
            
            # Update understanding with this batch
            understanding = self._update_understanding(
                current_understanding=understanding,
                new_events=batch_events,
                transcript_context=transcript_context,
                is_first_batch=(batch_idx == 0),
                is_last_batch=(batch_idx == num_batches - 1),
            )
        
        if verbose:
            self._log(f"Pass 2 complete: Built understanding with {len(understanding.steps)} steps", level="success")
            if understanding.task_goal:
                self._log(f"  Task goal: {understanding.task_goal[:80]}...")
            self._log(f"  Parameters detected: {len(understanding.parameters)}")
        
        return understanding
    
    def _update_understanding(
        self,
        current_understanding: RunningUnderstanding,
        new_events: list[DetectedEvent],
        transcript_context: str,
        is_first_batch: bool,
        is_last_batch: bool,
    ) -> RunningUnderstanding:
        """Update understanding with a new batch of events."""
        # Build message content
        content = self._build_understanding_message(
            current_understanding=current_understanding,
            new_events=new_events,
            transcript_context=transcript_context,
            is_first_batch=is_first_batch,
            is_last_batch=is_last_batch,
        )
        
        # Call the appropriate API
        if self.use_openai:
            response_text = self._call_openai_with_prompt(
                content, UNDERSTANDING_UPDATE_PROMPT, phase="pass2_understanding"
            )
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16384,  # Increased from 4096 to handle growing understanding
                system=UNDERSTANDING_UPDATE_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            response_text = response.content[0].text
            # Track usage
            self._track_usage(
                response.usage.input_tokens,
                response.usage.output_tokens,
                phase="pass2_understanding",
            )
        
        # Parse the updated understanding
        return self._parse_understanding_response(
            response_text, current_understanding
        )
    
    def _build_understanding_message(
        self,
        current_understanding: RunningUnderstanding,
        new_events: list[DetectedEvent],
        transcript_context: str,
        is_first_batch: bool,
        is_last_batch: bool,
    ) -> list[dict]:
        """Build message content for understanding update."""
        parts = []
        
        # Current state
        if is_first_batch:
            parts.append("# Initial Workflow Analysis\n\n")
            parts.append("This is the first batch of events. Start building the workflow understanding from scratch.\n\n")
        else:
            parts.append("# Workflow Understanding Update\n\n")
            parts.append("## Current Understanding\n\n")
            parts.append(f"```json\n{json.dumps(current_understanding.to_dict(), indent=2)}\n```\n\n")
        
        if is_last_batch:
            parts.append("**Note:** This is the final batch of events. Finalize the understanding.\n\n")
        
        # New events
        parts.append("## New Events to Process\n\n")
        events_data = [e.to_dict() for e in new_events]
        parts.append(f"```json\n{json.dumps(events_data, indent=2)}\n```\n\n")
        
        # Transcript context
        if transcript_context:
            parts.append("## Voice-Over Context\n\n")
            parts.append(f"{transcript_context}\n\n")
        
        parts.append("Update the understanding based on these new events. Output the complete updated understanding as JSON.")
        
        return [{"type": "text", "text": "".join(parts)}]
    
    def _parse_understanding_response(
        self,
        response_text: str,
        fallback: RunningUnderstanding,
    ) -> RunningUnderstanding:
        """Parse understanding JSON from model response."""
        data = extract_json_from_response(response_text, json_type="object", default=None)
        if data is not None:
            try:
                return RunningUnderstanding.from_dict(data)
            except KeyError as e:
                self._log(f"Failed to parse understanding response: {e}", level="warning")
        else:
            self._log(f"No valid JSON found in response (length: {len(response_text)} chars), falling back to previous understanding", level="warning")
        return fallback
    
    def _generate_workflow_pass(
        self,
        understanding: RunningUnderstanding,
        verbose: bool = True,
    ) -> Workflow:
        """Pass 3: Generate final workflow from understanding.
        
        Takes the complete accumulated understanding and synthesizes
        a polished markdown workflow document.
        
        Args:
            understanding: Complete RunningUnderstanding from Pass 2.
            verbose: Whether to print progress information.
            
        Returns:
            Final Workflow object.
        """
        if verbose:
            self._log("PASS 3: Workflow Synthesis", level="header")
            self._log(f"Synthesizing workflow from {len(understanding.steps)} steps...")
        
        # Build message content
        content = self._build_synthesis_message(understanding)
        
        # Use tqdm with a simple indeterminate progress (single iteration)
        with tqdm(total=1, desc="Generating workflow", unit="doc", disable=not verbose) as pbar:
            # Call the appropriate API
            if self.use_openai:
                response_text = self._call_openai_with_prompt(
                    content, WORKFLOW_SYNTHESIS_PROMPT, phase="pass3_synthesis"
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    system=WORKFLOW_SYNTHESIS_PROMPT,
                    messages=[{"role": "user", "content": content}],
                )
                response_text = response.content[0].text
                # Track usage
                self._track_usage(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    phase="pass3_synthesis",
                )
            pbar.update(1)
        
        # Parse the markdown response
        markdown_content = self._extract_markdown(response_text)
        workflow = Workflow.from_markdown(markdown_content)
        
        if verbose:
            self._log(f"Pass 3 complete!", level="success")
            self._log(f"  Workflow: {workflow.name}")
            self._log(f"  Parameters: {len(workflow.parameters)}")
        
        return workflow
    
    def _build_synthesis_message(
        self,
        understanding: RunningUnderstanding,
    ) -> list[dict]:
        """Build message content for workflow synthesis."""
        parts = []
        
        parts.append("# Complete Workflow Understanding\n\n")
        parts.append("Based on analyzing a screen recording, here is the complete understanding of the workflow:\n\n")
        parts.append(f"```json\n{json.dumps(understanding.to_dict(), indent=2)}\n```\n\n")
        parts.append("Please synthesize this into a comprehensive, polished markdown workflow document following the specified format.")
        
        return [{"type": "text", "text": "".join(parts)}]
