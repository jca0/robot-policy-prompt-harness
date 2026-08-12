import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

MODEL_ID = "gemini-robotics-er-1.6-preview"
TOPREWARD_MODEL_ID = "qwen.qwen3-vl-235b-a22b"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_PROMPTS_DIR = Path(__file__).parent / "prompts"
COMPLETION_PROMPT_TEMPLATE = (_PROMPTS_DIR / "completion_check.md").read_text()
MEMORY_WRITER_TEMPLATE = (_PROMPTS_DIR / "memory_writer.md").read_text()
NEXT_SUBTASK_PROMPT = (_PROMPTS_DIR / "next_subtask.md").read_text()
INITIAL_TEMPLATE_REACTIVE = (_PROMPTS_DIR / "initial_template_reactive.md").read_text()
