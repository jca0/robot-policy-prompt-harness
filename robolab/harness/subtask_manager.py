from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

from robolab.harness.constants import GOOGLE_API_KEY, MODEL_ID, NEXT_SUBTASK_PROMPT


@dataclass
class NextSubtaskResult:
    subtask: str | None
    done: bool
    reasoning: str | None = None


def get_next_subtask(
    goal: str,
    scene_frame: np.ndarray,
    memory: str,
    max_retries: int = 3,
) -> NextSubtaskResult:
    """Ask the VLM to decide the single next subtask given the current scene and memory."""
    client = genai.Client(api_key=GOOGLE_API_KEY)
    prompt = NEXT_SUBTASK_PROMPT.format(goal=goal, memory=memory)

    buf = io.BytesIO()
    Image.fromarray(scene_frame).save(buf, format="JPEG", quality=85)
    image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
    contents = [image_part, prompt]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[get_next_subtask] Attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    result = json.loads(text)
    return NextSubtaskResult(
        subtask=result.get("subtask"),
        done=bool(result.get("done", False)),
        reasoning=result.get("reasoning"),
    )
