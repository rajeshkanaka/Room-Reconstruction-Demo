"""
Gemini 3-based scene analysis.
Adds semantic understanding: room type, doors, windows, room shape.
Runs in parallel with VGGT -- does not slow down the pipeline.
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field
from termcolor import colored
from config import GEMINI_MODEL, GEMINI_CONFIDENCE_THRESHOLD


@dataclass
class SceneAnalysis:
    """Result of Gemini scene analysis."""

    success: bool = False
    room_type: str = ""
    room_shape: str = "rectangular"
    confidence: float = 0.0
    doors: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    features: list = field(default_factory=list)
    error: str = ""


ANALYSIS_PROMPT = """Analyze these room photos and return a JSON object with:

1. "room_type": one of "bedroom", "kitchen", "bathroom", "living_room", "dining_room", "office", "hallway", "other"
2. "room_shape": one of "rectangular", "l_shaped", "u_shaped", "irregular"
3. "confidence": your confidence in the analysis (0.0 to 1.0)
4. "doors": array of objects, each with:
   - "wall": which wall the door is on ("north", "south", "east", "west" -- use photo perspective)
   - "position": where on the wall ("left", "center", "right")
   - "width": estimated width ("narrow", "standard", "wide", "double")
   - "type": "interior", "exterior", "closet", "sliding"
5. "windows": array of objects, each with:
   - "wall": which wall
   - "position": where on the wall
   - "size": "small", "medium", "large", "floor_to_ceiling"
6. "features": array of notable architectural features seen

Be precise about door and window counts. Only report what you can clearly see."""

ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "room_type": {
            "type": "STRING",
            "enum": [
                "bedroom",
                "kitchen",
                "bathroom",
                "living_room",
                "dining_room",
                "office",
                "hallway",
                "other",
            ],
        },
        "room_shape": {
            "type": "STRING",
            "enum": ["rectangular", "l_shaped", "u_shaped", "irregular"],
        },
        "confidence": {"type": "NUMBER"},
        "doors": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "wall": {"type": "STRING"},
                    "position": {"type": "STRING"},
                    "width": {"type": "STRING"},
                    "type": {"type": "STRING"},
                },
            },
        },
        "windows": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "wall": {"type": "STRING"},
                    "position": {"type": "STRING"},
                    "size": {"type": "STRING"},
                },
            },
        },
        "features": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["room_type", "room_shape", "confidence", "doors", "windows"],
}


class SceneAnalyzer:
    """Analyze room photos using Gemini 3 for semantic understanding."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or GEMINI_MODEL
        self.client = None

    def _get_client(self):
        """Lazy-initialize the Gemini client via Vertex AI."""
        if self.client is not None:
            return self.client
        from google import genai

        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT", "adktalentpulse360"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
        return self.client

    def analyze(self, images: list[np.ndarray]) -> SceneAnalysis:
        """
        Analyze room photos for semantic content.

        Args:
            images: List of RGB images as numpy arrays (H, W, 3).

        Returns:
            SceneAnalysis with room type, doors, windows, shape.
        """
        try:
            from PIL import Image as PILImage
            from google.genai import types

            client = self._get_client()

            # Convert numpy arrays to PIL images
            pil_images = []
            for img in images[:5]:  # Limit to 5 images to control cost
                pil_images.append(PILImage.fromarray(img))

            # Build content: prompt text + images
            contents = [ANALYSIS_PROMPT] + pil_images

            print(
                colored(
                    f"[Gemini] Analyzing {len(pil_images)} images with {self.model_name}...",
                    "cyan",
                )
            )

            # Call Gemini with structured output
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ANALYSIS_SCHEMA,
                ),
            )

            data = json.loads(response.text)

            result = SceneAnalysis(
                success=True,
                room_type=data.get("room_type", "other"),
                room_shape=data.get("room_shape", "rectangular"),
                confidence=float(data.get("confidence", 0.0)),
                doors=data.get("doors", []),
                windows=data.get("windows", []),
                features=data.get("features", []),
            )

            print(
                colored(
                    f"[Gemini] Analysis: {result.room_type} ({result.room_shape}), "
                    f"{len(result.doors)} doors, {len(result.windows)} windows, "
                    f"confidence={result.confidence:.2f}",
                    "green",
                )
            )
            return result

        except Exception as e:
            print(colored(f"[Gemini] Scene analysis failed: {e}", "yellow"))
            return SceneAnalysis(success=False, error=str(e))
