# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Prompt calibration: learn which subtask prompt phrasings drive the most
task progress according to TOPReward, and bias future subtask generation
toward proven phrasings.
"""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

from robolab.harness.constants import GOOGLE_API_KEY, MODEL_ID, NEXT_SUBTASK_PROMPT
from robolab.harness.subtask_manager import NextSubtaskResult


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScoredPrompt:
    text: str
    reward_delta: float
    episode: int


@dataclass
class ScoredSequence:
    """A full subtask sequence from one episode, scored by outcome."""
    subtasks: list[str]
    goal_achieved: bool
    total_reward_delta: float
    total_steps: int
    episode: int


class CalibrationState:
    """Global pool of scored subtask sequences, persisted across episodes."""

    def __init__(self, state_path: Path, temperature: float = 0.8, context_k: int = 3):
        self._path = Path(state_path)
        self.temperature = temperature
        self.context_k = context_k
        self.prompts: list[ScoredPrompt] = []
        self.sequences: list[ScoredSequence] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            self.prompts = [ScoredPrompt(**p) for p in raw.get("prompts", [])]
            self.sequences = [ScoredSequence(**s) for s in raw.get("sequences", [])]
            self.temperature = raw.get("temperature", self.temperature)

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "temperature": self.temperature,
            "prompts": [{"text": p.text, "reward_delta": p.reward_delta, "episode": p.episode}
                        for p in self.prompts],
            "sequences": [
                {
                    "subtasks": s.subtasks,
                    "goal_achieved": s.goal_achieved,
                    "total_reward_delta": s.total_reward_delta,
                    "total_steps": s.total_steps,
                    "episode": s.episode,
                }
                for s in self.sequences
            ],
        }, indent=2))

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def add(self, prompt: ScoredPrompt):
        self.prompts.append(prompt)

    def add_sequence(self, sequence: ScoredSequence):
        self.sequences.append(sequence)

    def _aggregate(self) -> dict[str, float]:
        """Average reward_delta per unique prompt text."""
        totals: dict[str, list[float]] = {}
        for p in self.prompts:
            totals.setdefault(p.text, []).append(p.reward_delta)
        return {text: float(np.mean(deltas)) for text, deltas in totals.items()}

    # ------------------------------------------------------------------
    # Context for VLM injection
    # ------------------------------------------------------------------

    def format_context(self) -> str:
        lines: list[str] = []

        # Sequence-level context (primary signal)
        if self.sequences:
            lines.extend(self._format_sequence_context())

        # Fall back to individual prompt context if no sequences yet
        if not lines and self.prompts:
            lines.extend(self._format_prompt_context())

        return "\n".join(lines) if lines else ""

    def _format_sequence_context(self) -> list[str]:
        succeeded = [s for s in self.sequences if s.goal_achieved]
        failed = [s for s in self.sequences if not s.goal_achieved]

        lines = ["## Calibration History (learned strategies)"]

        if succeeded:
            # Rank by reward delta (higher = better), break ties by fewer steps
            succeeded.sort(key=lambda s: (-s.total_reward_delta, s.total_steps))
            lines.append("")
            lines.append("These subtask sequences SUCCEEDED (prefer similar strategies):")
            for seq in succeeded[: self.context_k]:
                steps_str = " → ".join(f'"{st}"' for st in seq.subtasks)
                lines.append(f"  ✓ {steps_str} (reward: {seq.total_reward_delta:+.3f}, steps: {seq.total_steps})")

        if failed:
            # Show worst failures
            failed.sort(key=lambda s: s.total_reward_delta)
            lines.append("")
            lines.append("These subtask sequences FAILED (avoid similar strategies):")
            for seq in failed[: self.context_k]:
                steps_str = " → ".join(f'"{st}"' for st in seq.subtasks)
                lines.append(f"  ✗ {steps_str} (reward: {seq.total_reward_delta:+.3f}, steps: {seq.total_steps})")

        if succeeded or failed:
            lines.append("")
            lines.append("Use this history to choose better ordering and approaches. "
                         "If an object is hard to grasp or obstructed, consider moving other objects first "
                         "or tackling easier objects before harder ones.")

        return lines

    def _format_prompt_context(self) -> list[str]:
        agg = self._aggregate()
        ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        k = self.context_k

        positive = [(text, delta) for text, delta in ranked if delta > 0]
        if not positive:
            return []

        lines = ["## Calibration History"]
        top = positive[:k]
        lines.append("Prefer prompts similar to these (high positive task progress):")
        for text, delta in top:
            lines.append(f'  - "{text}" (mean reward delta: {delta:+.3f})')

        bottom = ranked[-k:] if len(ranked) > k else []
        bottom = [(text, delta) for text, delta in bottom if delta <= 0]
        if bottom:
            lines.append("Avoid prompts similar to these (low or negative task progress):")
            for text, delta in bottom:
                lines.append(f'  - "{text}" (mean reward delta: {delta:+.3f})')

        return lines

    # ------------------------------------------------------------------
    # Output file
    # ------------------------------------------------------------------

    def write_ranked_results(self, path: Path) -> Path:
        """Write sequences and prompts ranked by effectiveness."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output: dict = {}

        # Sequences section
        if self.sequences:
            succeeded = sorted(
                [s for s in self.sequences if s.goal_achieved],
                key=lambda s: (-s.total_reward_delta, s.total_steps),
            )
            failed = sorted(
                [s for s in self.sequences if not s.goal_achieved],
                key=lambda s: s.total_reward_delta,
            )
            output["sequences"] = {
                "succeeded": [
                    {
                        "subtasks": s.subtasks,
                        "total_reward_delta": s.total_reward_delta,
                        "total_steps": s.total_steps,
                        "episode": s.episode,
                    }
                    for s in succeeded
                ],
                "failed": [
                    {
                        "subtasks": s.subtasks,
                        "total_reward_delta": s.total_reward_delta,
                        "total_steps": s.total_steps,
                        "episode": s.episode,
                    }
                    for s in failed
                ],
            }

        # Individual prompts section (legacy)
        if self.prompts:
            agg = self._aggregate()
            ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
            episodes_by_text: dict[str, list[dict]] = {}
            for p in self.prompts:
                episodes_by_text.setdefault(p.text, []).append(
                    {"episode": p.episode, "reward_delta": p.reward_delta}
                )
            output["prompts"] = [
                {
                    "rank": i + 1,
                    "text": text,
                    "mean_reward_delta": delta,
                    "n_episodes": len(episodes_by_text[text]),
                    "episodes": episodes_by_text[text],
                }
                for i, (text, delta) in enumerate(ranked)
            ]

        path.write_text(json.dumps(output, indent=2))
        return path


# ---------------------------------------------------------------------------
# Calibrated subtask proposer
# ---------------------------------------------------------------------------

def get_next_subtask_calibrated(
    goal: str,
    scene_frame: np.ndarray,
    memory: str,
    calibration_state: CalibrationState,
) -> NextSubtaskResult:
    """Like get_next_subtask but injects calibration history and uses learnt temperature."""
    client = genai.Client(api_key=GOOGLE_API_KEY)

    base_prompt = NEXT_SUBTASK_PROMPT.format(goal=goal, memory=memory)
    calib_context = calibration_state.format_context()
    prompt = f"{base_prompt}\n\n{calib_context}" if calib_context else base_prompt

    buf = io.BytesIO()
    Image.fromarray(scene_frame).save(buf, format="JPEG", quality=85)
    image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(temperature=calibration_state.temperature),
    )

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


# ---------------------------------------------------------------------------
# Reward delta helper
# ---------------------------------------------------------------------------

def reward_at_step(snapshots, target_step: int) -> float | None:
    """Return the raw_reward from the snapshot closest to target_step."""
    if not snapshots:
        return None
    return min(snapshots, key=lambda s: abs(s.step - target_step)).raw_reward
