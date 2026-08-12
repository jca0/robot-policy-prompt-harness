# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Barebones eval: run pi05 policy and record video. No VLM checker, no reward tracking.

Usage:
    python harness_scripts/run_eval_bare.py
    python harness_scripts/run_eval_bare.py --task Isaac-Franka-PickCoke-v0
    python harness_scripts/run_eval_bare.py --num-runs 5
    python harness_scripts/run_eval_bare.py --policy gr00t
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

parser = argparse.ArgumentParser(description="Barebones policy eval (no VLM)")

AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=None)
parser.add_argument("--policy",
                    choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05", "gr00t", "dreamzero", "molmo", "openvla", "openvla_oft"],
                    default="pi05")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1)
parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default")
parser.add_argument("--instruction", type=str, default=None,
                    help="Override the task's built-in instruction with a custom string.")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"])

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
args_cli.save_videos = args_cli.video_mode != "none"

# Defaults for variation flags expected by register_envs / build_eval_jobs
args_cli.backgrounds = None
args_cli.lighting_intensities = None
args_cli.lighting_types = None
args_cli.randomize_background = False
args_cli.background_seed = None
args_cli.table_materials = None
args_cli.camera_variations = None

EPISODE_LENGTH_S = 60

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
from harness_scripts.episode_bare import run_bare_episode

# If --task not provided, load from my_tasks.py
_my_tasks = None
if args_cli.task is None:
    from harness_scripts.my_tasks import TASKS as _my_tasks
    args_cli.task = [t["env"] for t in _my_tasks]

register_envs(args_cli)

# ===========================================================================
# Main
# ===========================================================================


def main():
    output_folder = get_timestamp() + "_" + args_cli.policy + "_bare"
    output_dir = os.path.join(PACKAGE_DIR, "output", output_folder)
    os.makedirs(output_dir, exist_ok=True)

    task_instructions = {}
    if _my_tasks is not None:
        task_instructions = {t["env"]: t["instruction"] for t in _my_tasks}

    jobs = build_eval_jobs(args_cli)
    num_runs = args_cli.num_runs

    print(f"\033[96m[Bare] {len(jobs)} jobs, {num_runs} episodes each, policy: {args_cli.policy}\033[0m")
    print(f"\033[96m[Bare] Output: {output_dir}\033[0m")

    episode_results_file, episode_results = init_experiment(output_dir)

    for job in jobs:
        scene_output_dir = os.path.join(output_dir, job.display_name)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(episode_results=episode_results,
                                       env_name=job.display_name,
                                       num_episodes=num_runs):
            print(f"\033[96m[Bare] {job.display_name} already done. Skipping.\033[0m")
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

        env_cfg.episode_length_s = EPISODE_LENGTH_S

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
                print(f"\033[96m[Bare] {job.display_name} run {run_idx} already done. Skipping.\033[0m")
                continue

            run_name = f"{job.display_name}_{run_idx}"
            ep_output_dir = os.path.join(scene_output_dir, "harness_logs", f"ep{run_idx}")
            os.makedirs(ep_output_dir, exist_ok=True)
            set_output_dir(ep_output_dir)

            env_results, msgs, timing = run_bare_episode(
                env=env, env_cfg=env_cfg, episode=run_idx, client=client,
                save_videos=args_cli.save_videos,
                video_mode=args_cli.video_mode,
                headless=args_cli.headless,
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
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[Bare] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
