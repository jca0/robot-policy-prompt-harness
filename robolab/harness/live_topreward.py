"""Live TOPReward tracking during simulation rollouts.

Accumulates frames in a buffer and scores trajectory prefixes in a
background thread so the sim loop is never blocked by Bedrock API calls.

Inlines the P(True) scoring logic from topreward.clients.bedrock_qwen so
we only need boto3 (available in the main venv) — no topreward dep.
"""

import base64
import io
import json
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from robolab.harness.constants import TOPREWARD_MODEL_ID


RESIZE_DIM = 244


def _frame_to_b64_jpeg(frame: np.ndarray) -> str:
    pil = Image.fromarray(frame).resize((RESIZE_DIM, RESIZE_DIM))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _score_frames(
    bedrock_client,
    model_name: str,
    frames: list[np.ndarray],
    instruction: str,
    max_images: int = 20,
) -> float:
    if len(frames) > max_images:
        indices = np.linspace(0, len(frames) - 1, max_images, dtype=int)
        frames = [frames[i] for i in indices]

    content: list[dict] = []
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_frame_to_b64_jpeg(frame)}"},
        })
    content.append({
        "type": "text",
        "text": (
            f"The above images show a robot manipulation trajectory that "
            f"completes the following task: {instruction}. "
            f"Decide whether the above statement is True or not. "
            f"Respond with only 'True' or 'False'."
        ),
    })

    body = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 10,
        "logprobs": True,
        "top_logprobs": 10,
    }
    response = bedrock_client.invoke_model(
        modelId=model_name,
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())

    choices = result.get("choices", [])
    if not choices:
        return -20.0
    logprobs_data = choices[0].get("logprobs", {})
    if not logprobs_data:
        return -20.0
    content_tokens = logprobs_data.get("content", [])
    if not content_tokens:
        return -20.0
    for entry in content_tokens[0].get("top_logprobs", []):
        if entry["token"].strip().lower() == "true":
            return float(entry["logprob"])
    raise RuntimeError("'true' token not found in top_logprobs")


@dataclass
class RewardSnapshot:
    step: int
    n_frames: int
    raw_reward: float
    timestamp: float
    subtask_index: int = 0


@dataclass
class SubtaskTransition:
    step: int
    subtask_index: int


class LiveRewardTracker:
    """Scores the trajectory-so-far in a background thread using TOPReward.

    Usage::

        tracker = LiveRewardTracker(instruction="put the cube in the bowl")
        tracker.start()

        for step in range(max_steps):
            frame = obs["policy"]["external_cam"][0].cpu().numpy()
            tracker.add_frame(frame, step)

        tracker.stop()
        tracker.save_plot(run_dir / "topreward_live.png")
    """

    def __init__(
        self,
        instruction: str,
        stride: int = 10,
        score_interval: float = 5.0,
        max_frames_per_call: int = 20,
        model_name: str = TOPREWARD_MODEL_ID,
        region_name: str = "us-west-2",
    ):
        import boto3

        self.instruction = instruction
        self.stride = stride
        self.score_interval = score_interval
        self.max_frames_per_call = max_frames_per_call
        self.model_name = model_name

        self._bedrock = boto3.client("bedrock-runtime", region_name=region_name)
        self._frames: list[np.ndarray] = []
        self._steps: list[int] = []
        self._lock = threading.Lock()
        self._results: list[RewardSnapshot] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._current_subtask_index: int = 0
        self._transitions: list[SubtaskTransition] = []

    def add_frame(self, frame: np.ndarray, step: int) -> None:
        with self._lock:
            self._frames.append(frame)
            self._steps.append(step)

    def set_subtask_index(self, index: int, step: int) -> None:
        self._current_subtask_index = index
        self._transitions.append(SubtaskTransition(step=step, subtask_index=index))

    def is_improving(self, window: int = 5) -> bool:
        """Check if TOPReward is monotonically increasing over the last `window` scores.

        Returns False if there are fewer than `window` scores for the current
        subtask, or if the scores are decreasing/oscillating.
        """
        subtask_scores = [
            s.raw_reward for s in self._results
            if s.subtask_index == self._current_subtask_index
        ]
        if len(subtask_scores) < window:
            return False
        recent = subtask_scores[-window:]
        return all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._scoring_loop, daemon=True)
        self._thread.start()
        print("LiveRewardTracker started")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=30)
        self._score_once()
        print(f"LiveRewardTracker stopped, {len(self._results)} snapshots collected")

    def reset(self) -> None:
        with self._lock:
            self._frames.clear()
            self._steps.clear()
        self._results.clear()
        self._current_subtask_index = 0
        self._transitions.clear()

    @property
    def snapshots(self) -> list[RewardSnapshot]:
        return list(self._results)

    def _scoring_loop(self) -> None:
        while self._running:
            self._score_once()
            time.sleep(self.score_interval)

    def _score_once(self) -> None:
        with self._lock:
            if len(self._frames) < 2:
                return
            frames = list(self._frames)
            steps = list(self._steps)

        sampled = self._subsample(frames)
        try:
            reward = _score_frames(
                self._bedrock,
                self.model_name,
                sampled,
                self.instruction,
                max_images=self.max_frames_per_call,
            )
            snapshot = RewardSnapshot(
                step=steps[-1],
                n_frames=len(frames),
                raw_reward=reward,
                timestamp=time.time(),
                subtask_index=self._current_subtask_index,
            )
            self._results.append(snapshot)
            print(
                f"TOPReward step={snapshot.step} "
                f"frames={snapshot.n_frames} "
                f"reward={snapshot.raw_reward:.4f}"
            )
        except Exception:
            traceback.print_exc()

    def _subsample(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        strided = frames[:: self.stride]
        if len(strided) <= self.max_frames_per_call:
            return strided
        indices = np.linspace(0, len(strided) - 1, self.max_frames_per_call, dtype=int)
        return [strided[i] for i in indices]

    @staticmethod
    def _normalize(rewards: list[float]) -> list[float]:
        arr = np.array(rewards, dtype=np.float64)
        if len(arr) <= 1:
            return [1.0] * len(arr)
        r_min, r_max = arr.min(), arr.max()
        if r_max == r_min:
            return [1.0] * len(arr)
        return ((arr - r_min) / (r_max - r_min)).tolist()

    _SUBTASK_COLORS = [
        "#E53935",  # red
        "#43A047",  # green
        "#1E88E5",  # blue
        "#FB8C00",  # orange
        "#8E24AA",  # purple
        "#00ACC1",  # cyan
        "#FFB300",  # amber
        "#6D4C41",  # brown
        "#D81B60",  # pink
        "#3949AB",  # indigo
    ]

    def save_plot(self, path: Path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._results:
            print("WARNING: No reward snapshots to plot")
            return path

        steps = [s.step for s in self._results]
        rewards = [s.raw_reward for s in self._results]
        normalized = self._normalize(rewards)
        subtask_indices = [s.subtask_index for s in self._results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        self._plot_color_coded(ax1, steps, rewards, subtask_indices)
        ax1.set_xlabel("Simulation Step")
        ax1.set_ylabel("log P(True)")
        ax1.set_title("Raw TOPReward")
        ax1.grid(True, alpha=0.3)

        self._plot_color_coded(ax2, steps, normalized, subtask_indices)
        ax2.set_xlabel("Simulation Step")
        ax2.set_ylabel("Normalized Reward")
        ax2.set_title("Normalized TOPReward")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)

        legend_indices = sorted(set(subtask_indices))
        handles = [
            plt.Line2D([0], [0], color=self._SUBTASK_COLORS[idx % len(self._SUBTASK_COLORS)],
                       linewidth=2, marker="o", label=f"Subtask {idx + 1}")
            for idx in legend_indices
        ]
        ax2.legend(handles=handles, loc="upper left", fontsize=8)

        title = self.instruction if len(self.instruction) <= 60 else self.instruction[:57] + "..."
        fig.suptitle(f'TOPReward: "{title}"', fontsize=10, fontweight="bold")
        plt.tight_layout()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Live reward plot saved to {path}")
        return path

    def _plot_color_coded(
        self,
        ax,
        steps: list[int],
        values: list[float],
        subtask_indices: list[int],
    ) -> None:
        for i in range(len(steps) - 1):
            color = self._SUBTASK_COLORS[subtask_indices[i] % len(self._SUBTASK_COLORS)]
            ax.plot(
                steps[i : i + 2], values[i : i + 2],
                color=color, linewidth=2,
            )
        for i, (x, y) in enumerate(zip(steps, values)):
            color = self._SUBTASK_COLORS[subtask_indices[i] % len(self._SUBTASK_COLORS)]
            ax.plot(x, y, marker="o", color=color, markersize=5)

    def save_subtask_plots(self, output_dir: Path, subtask_log: list[dict]) -> None:
        """Score and plot TOPReward for each individual subtask.

        For each subtask, scores frames from that subtask's time range using
        the subtask instruction as the query, then saves the plot.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not subtask_log or not self._frames:
            return

        topreward_dir = Path(output_dir) / "topreward"
        topreward_dir.mkdir(parents=True, exist_ok=True)

        # Move the main plot into the topreward folder too
        main_plot = output_dir / "topreward.png"
        if main_plot.exists():
            main_plot.rename(topreward_dir / "topreward.png")
        main_jsonl = output_dir / "topreward.jsonl"
        if main_jsonl.exists():
            main_jsonl.rename(topreward_dir / "topreward.jsonl")

        with self._lock:
            all_frames = list(self._frames)
            all_steps = list(self._steps)

        # Map subtask_index (0-based) to its start step from transitions
        idx_to_start = {t.subtask_index: t.step for t in self._transitions}

        for entry in subtask_log:
            subtask_idx = entry["index"]  # 1-based
            subtask_text = entry["subtask"]
            status = entry["status"]

            # Find the start/end steps for this subtask
            zero_idx = subtask_idx - 1  # convert to 0-based
            start_step = idx_to_start.get(zero_idx, 0)
            # End step is the start of the next subtask, or last frame
            next_idx = zero_idx + 1
            if next_idx in idx_to_start:
                end_step = idx_to_start[next_idx]
            else:
                end_step = all_steps[-1] if all_steps else 0

            # Get frames within this subtask's step range
            subtask_frames = []
            subtask_steps = []
            for i, s in enumerate(all_steps):
                if start_step <= s <= end_step:
                    subtask_frames.append(all_frames[i])
                    subtask_steps.append(s)

            if len(subtask_frames) < 2:
                continue

            # Score increasing prefixes of the subtask frames
            snapshots = []
            n_scores = min(8, len(subtask_frames))
            score_indices = np.linspace(1, len(subtask_frames) - 1, n_scores, dtype=int)

            for idx in score_indices:
                prefix = subtask_frames[: idx + 1]
                sampled = prefix[:: self.stride] if len(prefix) > self.max_frames_per_call else prefix
                if len(sampled) > self.max_frames_per_call:
                    sample_idx = np.linspace(0, len(sampled) - 1, self.max_frames_per_call, dtype=int)
                    sampled = [sampled[i] for i in sample_idx]
                try:
                    reward = _score_frames(
                        self._bedrock,
                        self.model_name,
                        sampled,
                        subtask_text,
                        max_images=self.max_frames_per_call,
                    )
                    snapshots.append((subtask_steps[idx], reward))
                except Exception:
                    traceback.print_exc()
                    continue

            if not snapshots:
                continue

            # Plot
            steps_plot = [s[0] for s in snapshots]
            rewards_plot = [s[1] for s in snapshots]
            normalized = self._normalize(rewards_plot)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            color = self._SUBTASK_COLORS[(subtask_idx - 1) % len(self._SUBTASK_COLORS)]

            ax1.plot(steps_plot, rewards_plot, color=color, linewidth=2, marker="o", markersize=5)
            ax1.set_xlabel("Simulation Step")
            ax1.set_ylabel("log P(True)")
            ax1.set_title("Raw TOPReward")
            ax1.grid(True, alpha=0.3)

            ax2.plot(steps_plot, normalized, color=color, linewidth=2, marker="o", markersize=5)
            ax2.set_xlabel("Simulation Step")
            ax2.set_ylabel("Normalized Reward")
            ax2.set_title("Normalized TOPReward")
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, alpha=0.3)

            title_text = subtask_text if len(subtask_text) <= 50 else subtask_text[:47] + "..."
            fig.suptitle(
                f'Subtask {subtask_idx}: "{title_text}" [{status}]',
                fontsize=10, fontweight="bold",
            )
            plt.tight_layout()

            plot_path = topreward_dir / f"topreward_subtask{subtask_idx}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Subtask TOPReward plot saved to {plot_path}")

    def save_jsonl(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for s in self._results:
                f.write(json.dumps({
                    "step": s.step,
                    "n_frames": s.n_frames,
                    "raw_reward": s.raw_reward,
                    "timestamp": s.timestamp,
                    "subtask_index": s.subtask_index,
                }) + "\n")
        return path
