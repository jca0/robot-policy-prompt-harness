import io
import json
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

from robolab.harness.constants import GOOGLE_API_KEY, MODEL_ID, MEMORY_WRITER_TEMPLATE, INITIAL_TEMPLATE_REACTIVE


class MemoryManager:
    """Maintains a memory.md file that tracks scene state across subtasks.

    Injects memory into the completion checker so repeated subtasks are
    handled correctly (e.g. distinguishing '1 cube in bowl' vs '2 cubes in bowl').
    """

    def __init__(self, memory_path: Path, model_id: str = MODEL_ID):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_id = model_id
        self.memory_path = memory_path
        self._memory: str = ""
        self._completed_subtasks: list[str] = []
        self._failed_subtasks: list[dict] = []
        self._last_frame: np.ndarray | None = None

    def reset(self, goal: str, initial_frame: np.ndarray | None = None):
        """Initialize memory at the start of a new episode (reactive mode, no fixed plan)."""
        initial_scene = self._describe_initial_scene(initial_frame, goal) if initial_frame is not None else "(no image available)"
        self._memory = INITIAL_TEMPLATE_REACTIVE.format(goal=goal, initial_scene=initial_scene)
        self._completed_subtasks = []
        self._failed_subtasks = []
        self._last_frame = initial_frame.copy() if initial_frame is not None else None
        self._write()

    def get_memory(self) -> str:
        return self._memory

    def last_frame(self) -> np.ndarray | None:
        return self._last_frame

    def context_for(self, subtask: str, subtask_index: int) -> str:
        prior_count = self._completed_subtasks.count(subtask)
        warning = ""
        if prior_count > 0:
            warning = (
                f"\n⚠ WARNING: This exact subtask has already been completed "
                f"{prior_count} time(s). Do NOT mark it complete just because "
                f"the action from a previous run of this subtask is visible — "
                f"the scene must show additional progress beyond what is described "
                f"in 'Scene Before This Subtask' above.\n"
            )

        header = f"## Current Subtask Context\nSubtask {subtask_index + 1}: \"{subtask}\"{warning}"
        return self._memory + "\n" + header

    def update(self, frame: np.ndarray, completed_subtask: str):
        changes = self._get_scene_diff(frame, completed_subtask)

        entry = f"- {completed_subtask}: {changes}"
        self._memory = self._memory.replace(
            "(none yet)", entry
        ) if "(none yet)" in self._memory else self._memory + "\n" + entry

        self._completed_subtasks.append(completed_subtask)
        self._last_frame = frame.copy()
        self._write()

    def record_timeout(self, frame: np.ndarray, failed_subtask: str):
        """Record a subtask that timed out without completion."""
        changes = self._get_scene_diff(frame, failed_subtask)
        attempt_num = sum(1 for f in self._failed_subtasks if f["subtask"] == failed_subtask) + 1
        self._failed_subtasks.append({"subtask": failed_subtask, "attempt": attempt_num})

        failed_section_header = "\n\n## Failed Subtasks (timed out)"
        entry = f"- {failed_subtask}: {changes} (attempt {attempt_num})"

        if failed_section_header.strip() in self._memory:
            self._memory += "\n" + entry
        else:
            self._memory += failed_section_header + "\n" + entry

        self._last_frame = frame.copy()
        self._write()

    def _get_scene_diff(self, after_frame: np.ndarray, subtask: str) -> str:
        prompt = MEMORY_WRITER_TEMPLATE.format(subtask=subtask)

        after_part = types.Part.from_bytes(
            data=_image_bytes(after_frame), mime_type="image/jpeg",
        )

        if self._last_frame is not None:
            before_part = types.Part.from_bytes(
                data=_image_bytes(self._last_frame), mime_type="image/jpeg",
            )
            contents = [before_part, after_part, prompt]
        else:
            contents = [after_part, prompt]

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0),
        )

        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        try:
            return json.loads(text).get("changes", text)
        except json.JSONDecodeError:
            return text

    def _describe_initial_scene(self, frame: np.ndarray, goal: str) -> str:
        image_part = types.Part.from_bytes(data=_image_bytes(frame), mime_type="image/jpeg")

        prompt = (
            f"Describe the current scene state for objects relevant to this task: \"{goal}\". "
            "Be short and concrete (e.g. \"red cube on table to the left of bowl, bowl is empty\"). "
            "Only mention task-relevant objects."
        )
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response.text.strip() if response.text else "(unable to describe scene)"

    def _write(self):
        self.memory_path.write_text(self._memory)


def _image_bytes(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=85)
    return buf.getvalue()
