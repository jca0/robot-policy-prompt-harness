import io
import json
import os

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

from robolab.harness.constants import GOOGLE_API_KEY, MODEL_ID, COMPLETION_PROMPT_TEMPLATE


class ProgressMonitor:
    """Calls a VLM every N steps to check if the current subtask is done."""

    def __init__(
        self,
        check_every_n_steps: int = 15,
        model_id: str = MODEL_ID,
    ):
        api_key = GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.check_every_n_steps = check_every_n_steps
        self._latest_frame: np.ndarray | None = None

    def reset(self):
        self._latest_frame = None

    def set_frame(self, frame: np.ndarray):
        """Store the current camera frame (H, W, 3 uint8 RGB)."""
        self._latest_frame = frame

    def check_completion(self, subtask: str, memory: str = "", before_frame: np.ndarray | None = None) -> dict:
        prompt = COMPLETION_PROMPT_TEMPLATE.format(subtask=subtask, memory=memory)

        current_part = types.Part.from_bytes(data=_image_to_bytes(Image.fromarray(self._latest_frame)), mime_type="image/jpeg")

        if before_frame is not None:
            before_part = types.Part.from_bytes(data=_image_to_bytes(Image.fromarray(before_frame)), mime_type="image/jpeg")
            contents = [before_part, current_part, prompt]
        else:
            contents = [current_part, prompt]

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0),
        )

        raw_response = response.text
        result = _parse_response(raw_response)
        print(f"Subtask result: {result}")
        result["raw_vlm_response"] = raw_response
        result["prompt_sent"] = prompt
        result["frame"] = self._latest_frame.copy()
        return result


def _image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    try:
        result = json.loads(text)
        return {
            "completed": bool(result.get("completed", False)),
            "reason": str(result.get("reason", "")),
        }
    except json.JSONDecodeError:
        return {"completed": False, "reason": f"Failed to parse VLM response: {text}"}
