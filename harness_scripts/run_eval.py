# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Modular evaluation with VLM-based termination.

Usage:
    python harness_scripts/run_eval.py 
    python harness_scripts/run_eval.py --decomposition
    python harness_scripts/run_eval.py --calibrate --num-runs 10
    python harness_scripts/run_eval.py --calibrate --calibration-temperature 0.5 --calibration-context-k 8

If --task is omitted, tasks are loaded from harness_scripts/my_tasks.py.
"""

import argparse
import cv2  # Must import before isaaclab
import os
import sys
import traceback

from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Modular evaluation with VLM termination")

AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=None)
parser.add_argument("--policy",
                    choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05", "gr00t", "dreamzero", "molmo", "openvla", "openvla_oft"],
                    default="pi05")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1)
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")
parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default") # default, vague, specific
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"])

parser.add_argument("--randomize-background", "--randomize_background", action="store_true")
parser.add_argument("--background-seed", "--background_seed", type=int, default=None)

# Instruction override
parser.add_argument("--instruction", type=str, default=None,
                    help="Override the task's built-in instruction with a custom string.")

# Variation arguments
var_group = parser.add_argument_group("Variation arguments")
var_group.add_argument("--backgrounds", nargs='+', default=None)
var_group.add_argument("--lighting-intensities", "--lighting_intensities", nargs='+', type=int, default=None)
var_group.add_argument("--lighting-types", "--lighting_types", nargs='+', default=None,
                       choices=["directional", "red", "blue", "green"])
var_group.add_argument("--table-materials", "--table_materials", nargs='+', default=None)
var_group.add_argument("--camera-variations", "--camera_variations", nargs='+', default=None,
                       choices=["external", "wrist", "both"])

# Decomposition / harness arguments
dp_group = parser.add_argument_group("Decomposition arguments")
dp_group.add_argument("--decomposition", "--decompose", action="store_true")
dp_group.add_argument("--check-every-n-steps", "--check_every_n_steps", type=int, default=45)
dp_group.add_argument("--subtask-timeout-steps", "--subtask_timeout_steps", type=int, default=300,
                      help="Steps before a subtask is marked timed_out and reprompted (default: 450)")
dp_group.add_argument("--topreward-stride", "--topreward_stride", type=int, default=10)
dp_group.add_argument("--topreward-interval", "--topreward_interval", type=float, default=5.0)

# Calibration arguments
cal_group = parser.add_argument_group("Calibration arguments")
cal_group.add_argument("--calibrate", action="store_true", default=False,
                       help="Run calibration mode: score subtask prompts with TOPReward delta.")
cal_group.add_argument("--calibration-temperature", "--calibration_temperature",
                       type=float, default=0.8,
                       help="VLM temperature for subtask generation (0.0=exploit, 1.0+=explore).")
cal_group.add_argument("--calibration-context-k", "--calibration_context_k",
                       type=int, default=5,
                       help="Number of top/bottom scored prompts to inject as few-shot context.")

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
args_cli.save_videos = args_cli.video_mode != "none"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===========================================================================
# Post-sim imports
# ===========================================================================

import robolab.constants
from robolab.constants import PACKAGE_DIR, set_output_dir
from robolab.eval import create_client, summarize_run
from robolab.core.environments.runtime import create_env
from robolab.core.logging.results import (
    init_experiment, check_all_episodes_complete,
    check_run_complete, summarize_experiment_results,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness_scripts.variations import register_envs, build_eval_jobs, change_table_material
from harness_scripts.episode_vlm import run_vlm_episode
from harness_scripts.episode_dynamic import run_dynamic_prompting_episode
from harness_scripts.episode_calibration import run_calibration_episode
from harness_scripts.prompt_calibration import CalibrationState

robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data

# If --task not provided on CLI, load from my_tasks.py
_my_tasks = None
if args_cli.task is None:
    from harness_scripts.my_tasks import TASKS as _my_tasks
    args_cli.task = [t["env"] for t in _my_tasks]

register_envs(args_cli)

# ===========================================================================
# Main
# ===========================================================================


def main():
    suffix_parts = [args_cli.policy]
    if args_cli.calibrate:
        suffix_parts.append("calibrate")
    elif args_cli.decomposition:
        suffix_parts.append("decompose")
    if args_cli.backgrounds:
        suffix_parts.append("bg")
    if args_cli.lighting_intensities or args_cli.lighting_types:
        suffix_parts.append("lighting")
    if args_cli.table_materials:
        suffix_parts.append("table")
    if args_cli.camera_variations:
        suffix_parts.append("camera")
    if args_cli.instruction:
        suffix_parts.append("custom")
    if args_cli.instruction_type != "default":
        suffix_parts.append(args_cli.instruction_type)
    output_folder = get_timestamp() + "_" + "_".join(suffix_parts)

    output_dir = os.path.join(PACKAGE_DIR, "output", output_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Build per-env instruction map from my_tasks.py if loaded
    task_instructions = {}
    if _my_tasks is not None:
        task_instructions = {t["env"]: t["instruction"] for t in _my_tasks}

    jobs = build_eval_jobs(args_cli)
    num_runs = args_cli.num_runs
    total_episodes = num_runs

    if args_cli.calibrate:
        mode = "calibration"
    elif args_cli.decomposition:
        mode = "decomposition"
    else:
        mode = "VLM termination"
    print(f"\033[96m[Harness] {len(jobs)} jobs, {total_episodes} episodes each, mode: {mode}\033[0m")
    print(f"\033[96m[Harness] Output: {output_dir}\033[0m")
    if args_cli.instruction:
        print(f"\033[96m[Harness] Instruction override: \"{args_cli.instruction}\"\033[0m")

    episode_results_file, episode_results = init_experiment(output_dir)

    calibration_state = None
    if args_cli.calibrate:
        calibration_state = CalibrationState(
            state_path=os.path.join(output_dir, "calibration_state.json"),
            temperature=args_cli.calibration_temperature,
            context_k=args_cli.calibration_context_k,
        )
        print(f"\033[96m[Harness] Calibration: temp={args_cli.calibration_temperature}, "
              f"context_k={args_cli.calibration_context_k}\033[0m")

    for job in jobs:
        scene_output_dir = os.path.join(output_dir, job.display_name)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(episode_results=episode_results,
                                       env_name=job.display_name,
                                       num_episodes=total_episodes):
            print(f"\033[96m[Harness] {job.display_name} already done. Skipping.\033[0m")
            continue

        env, env_cfg = create_env(
            job.base_task_env,
            device=args_cli.device,
            num_envs=1,
            use_fabric=True,
            events=job.camera_event,
            instruction_type=args_cli.instruction_type,
            policy=args_cli.policy,
        )

        # Instruction priority: --instruction flag > my_tasks.py > task built-in
        if args_cli.instruction is not None:
            env_cfg.instruction = args_cli.instruction
        elif job.base_task_env in task_instructions:
            env_cfg.instruction = task_instructions[job.base_task_env]

        if job.table_material is not None:
            change_table_material(job.table_material)

        client = create_client(args_cli.policy)

        for run_idx in range(num_runs):
            if check_run_complete(episode_results=episode_results,
                                  env_name=job.display_name, episode=run_idx):
                print(f"\033[96m[Harness] {job.display_name} run {run_idx} already done. Skipping.\033[0m")
                continue

            run_name = f"{job.display_name}_{run_idx}"
            ep_output_dir = os.path.join(scene_output_dir, "harness_logs", f"ep{run_idx}")
            os.makedirs(ep_output_dir, exist_ok=True)
            set_output_dir(ep_output_dir)

            print(f"\033[96m[Harness] Running {run_name}: '{env_cfg.instruction}' (run {run_idx})\033[0m")

            from robolab.harness.live_topreward import LiveRewardTracker
            reward_tracker = LiveRewardTracker(
                instruction=env_cfg.instruction,
                stride=args_cli.topreward_stride,
                score_interval=args_cli.topreward_interval,
            )
            reward_tracker.start()

            if args_cli.calibrate:
                env_results, msgs, timing = run_calibration_episode(
                    env=env, env_cfg=env_cfg, episode=run_idx, client=client,
                    calibration_state=calibration_state,
                    save_videos=args_cli.save_videos,
                    video_mode=args_cli.video_mode,
                    headless=args_cli.headless,
                    check_every_n_steps=args_cli.check_every_n_steps,
                    subtask_timeout_steps=args_cli.subtask_timeout_steps,
                    reward_tracker=reward_tracker,
                    scene_output_dir=scene_output_dir,
                )
            elif args_cli.decomposition:
                env_results, msgs, timing = run_dynamic_prompting_episode(
                    env=env, env_cfg=env_cfg, episode=run_idx, client=client,
                    save_videos=args_cli.save_videos,
                    video_mode=args_cli.video_mode,
                    headless=args_cli.headless,
                    check_every_n_steps=args_cli.check_every_n_steps,
                    subtask_timeout_steps=args_cli.subtask_timeout_steps,
                    reward_tracker=reward_tracker,
                    scene_output_dir=scene_output_dir,
                )
            else:
                env_results, msgs, timing = run_vlm_episode(
                    env=env, env_cfg=env_cfg, episode=run_idx, client=client,
                    save_videos=args_cli.save_videos,
                    video_mode=args_cli.video_mode,
                    headless=args_cli.headless,
                    check_every_n_steps=args_cli.check_every_n_steps,
                    reward_tracker=reward_tracker,
                    scene_output_dir=scene_output_dir,
                )

            episode_results = summarize_run(
                env_results=env_results,
                msgs=msgs,
                timing=timing,
                env=env,
                env_cfg=env_cfg,
                num_envs=1,
                run_idx=run_idx,
                run_name=run_name,
                task_env=job.display_name,
                scene_output_dir=ep_output_dir,
                policy=args_cli.policy,
                instruction_type=args_cli.instruction_type,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
                enable_subtask_progress=False,
                task_name=job.task_name,
                extra_fields=job.extra_fields,
            )

            env.reset_eval_state()

        env.close()

    summarize_experiment_results(episode_results, show_timing=True)

    if calibration_state is not None:
        ranked_path = calibration_state.write_ranked_results(
            Path(output_dir) / "calibration_ranked.json"
        )
        print(f"\033[96m[Harness] Calibration results: {ranked_path}\033[0m")

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[Harness] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
