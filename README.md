# Robot Policy Prompt Harness

A harness for **dynamic prompt optimization of robot manipulation policies**, built on top of [NVIDIA RoboLab](https://github.com/NVlabs/RoboLab). The premise: the language instruction a policy receives is a free variable you can optimize — at runtime and across runs — rather than a fixed label on the task. The harness optimizes it on two timescales:

- **Within an episode**, a VLM adapts the prompt on the fly: it proposes the next subtask from the live camera feed, monitors progress, keeps a running scene memory, and re-prompts the policy at each subtask boundary while a reward model ([TOPReward](https://github.com/jca0/TOPReward)) scores progress in the background.
- **Across episodes**, calibration mode treats subtask prompts and their orderings as candidates to be searched: it explores variations, scores each against the reward signal, and feeds the best and worst strategies back into future proposals.

The harness is policy-agnostic: it treats the policy as a black box that takes an image observation and a text instruction and returns actions. VLAs like Pi0.5, GR00T, and OpenVLA are the typical case, but anything with an image+text interface works — the harness only ever changes the instruction string it sends.

## How it works

Each episode runs a closed loop around the policy (default: Pi0.5 via a RoboLab inference client):

1. **Scene description** — Gemini (`gemini-robotics-er-1.6-preview`) describes the initial scene into a persistent `memory.md`.
2. **Reactive subtask proposal** — given the goal, the current frame, and the memory doc, the VLM proposes *one* next subtask (`robolab/harness/subtask_manager.py`). There is no up-front plan; the decomposition is re-decided after every subtask.
3. **Policy execution** — the policy is prompted with the current subtask text on every inference step.
4. **Completion checking** — every N steps, the VLM compares before/current frames and judges whether the subtask is done (`robolab/harness/progress_monitor.py`).
5. **Memory writing** — on subtask completion, the VLM writes a scene diff into memory; timed-out subtasks are recorded as failures so they aren't blindly retried (`robolab/harness/memory_manager.py`).
6. **Live reward tracking** — a background thread scores the trajectory-so-far with TOPReward (Qwen3-VL-235B on AWS Bedrock, P("True") logprob as the reward signal). If a subtask times out but the reward trend is still improving, its deadline is extended (`robolab/harness/live_topreward.py`).
7. The episode ends when the VLM declares the overall goal achieved.

### Calibration mode

Calibration mode is the cross-episode half of the optimization: it learns *which subtask prompts and orderings actually work* for a task:

- Subtask proposals are sampled at higher temperature (exploration).
- Each completed subtask prompt is scored by its **TOPReward delta** (reward gained while the prompt was active).
- Each finished episode records the full subtask **sequence** with its outcome, total reward delta, and step count.
- Accumulated results persist in `calibration_state.json` across runs, and the top/bottom-k sequences are injected back into the next-subtask prompt as "prefer"/"avoid" strategies — so later episodes exploit what earlier ones learned.

Final rankings are written to `calibration_ranked.json`.

## Repository layout

| Path | What it is |
| --- | --- |
| `harness_scripts/` | Entry points and episode runners (this repo's core addition) |
| `robolab/harness/` | Harness library: subtask manager, progress monitor, memory manager, live TOPReward tracker, prompt templates |
| `topreward/` | [TOPReward](https://github.com/jca0/TOPReward) submodule (reference implementation; the harness inlines its Bedrock client and doesn't import it at runtime) |
| `examples/policy/run_dynamic_prompting.py` | Earlier standalone version of the dynamic-prompting runner |
| `info.md` | Scene → task → instruction table for all evaluation tasks |
| everything else | Stock RoboLab (tasks, assets, sim infrastructure) — see [docs/ROBOLAB_README.md](docs/ROBOLAB_README.md) |

## Setup

Follow the [RoboLab installation](docs/ROBOLAB_README.md) (uv + Isaac Sim 5.0 / Isaac Lab 2.2.0, installed via `uv sync`), then configure credentials:

```bash
# .env in the repo root
GOOGLE_API_KEY=...        # Gemini, for subtask proposal / completion checks / memory
```

TOPReward scoring additionally needs standard AWS credentials with Bedrock access in `us-west-2` (model: `qwen.qwen3-vl-235b-a22b`).

Your policy must be running as a RoboLab-compatible inference server (OpenPI, GR00T, etc. — see [docs/inference.md](docs/inference.md)). Any policy that accepts an image observation and a text instruction can be wrapped this way; it doesn't have to be a VLA.

## Usage

All runs go through `harness_scripts/run_eval.py` (cameras and headless mode are forced on):

```bash
# VLM-termination mode: fixed instruction, VLM only decides when the episode is done
uv run python harness_scripts/run_eval.py --policy pi05

# Dynamic subtask decomposition (the full loop described above)
uv run python harness_scripts/run_eval.py --decomposition --task BlocksInBinTask

# Prompt calibration across repeated runs
uv run python harness_scripts/run_eval.py --calibrate --num-runs 10 --task BlocksInBinTask

# Bare baseline: policy + video only, no VLM
uv run python harness_scripts/run_eval_bare.py --task BlocksInBinTask
```

Useful flags: `--policy` (pi0, pi05, gr00t, openvla, …), `--task` (one or more; defaults to the list in `harness_scripts/my_tasks.py`), `--instruction` (override), `--check-every-n-steps` (completion-check cadence, default 45), `--subtask-timeout-steps` (default 300), `--topreward-stride` / `--topreward-interval` (reward sampling), `--calibration-temperature` / `--calibration-context-k`, plus RoboLab's scene variation flags (`--backgrounds`, `--lighting-types`, `--table-materials`, `--camera-variations`).

## Output artifacts

Per episode, under `<output>/<task>/harness_logs/ep<N>/`:

- `memory.md` — the VLM's running scene memory (initial scene, completed subtasks, failures)
- `subtasks.json` — goal, outcome, and every subtask with status (`succeeded` / `timed_out` / `abandoned`) and steps taken
- `harness.log` — full harness event log
- `topreward/` — reward curve plot + JSONL, plus per-subtask reward plots

Calibration runs additionally produce `calibration_state.json` (persistent, cross-run) and `calibration_ranked.json` (final ranked prompts and sequences).

## Acknowledgments

Built on [RoboLab](https://github.com/NVlabs/RoboLab) by NVIDIA (CC-BY-NC-4.0) — the simulation benchmark, tasks, and assets are theirs; see the [original README](docs/ROBOLAB_README.md). Reward scoring uses [TOPReward](https://github.com/jca0/TOPReward).
