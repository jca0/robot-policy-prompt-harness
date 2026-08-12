# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Run policy evaluation with reactive dynamic prompting.

Instead of feeding a single instruction for the entire episode, a VLM
decomposes the task on the fly: it picks the next subtask based on the
current scene observation, monitors progress, and maintains memory of
what has been completed. The policy receives the current subtask as its
instruction at each step.

Supports multi-env: each "run" spawns num_envs parallel episodes.
Total episodes = num_runs * num_envs.

Usage:
    Run on all registered tasks:
    $ python run_dynamic_prompting.py

    Run on specific tasks:
    $ python run_dynamic_prompting.py --task BananaInBowlTask

    Use specific policy:
    $ python run_dynamic_prompting.py --policy pi05

    Adjust VLM check frequency:
    $ python run_dynamic_prompting.py --check-every-n-steps 20

Output:
    Results are saved to: output/<output_folder_name>/
"""

import argparse
import cv2  # Must import this before isaaclab. Do not remove
import os
import traceback
import sys
from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp, DEFAULT_TASK_SUBFOLDERS  # noqa

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=None,
                       help="List of tasks to evaluate on ")
parser.add_argument("--tag", nargs='+', default=None,
                       help="List of tags of tasks to evaluate on ")
parser.add_argument("--task-dirs", nargs='+', default=DEFAULT_TASK_SUBFOLDERS,
                       help="List of task directories to evaluate on")
parser.add_argument("--policy",
                    choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05", "gr00t", "dreamzero", "molmo", "openvla", "openvla_oft"], default="pi05",
                       help="Action-prediction backend to use (default: pi05)")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                       help="Number of sequential runs per task (default: 1). Total episodes = num_runs * num_envs.")
parser.add_argument("--enable-subtask", "--enable_subtask", action="store_true",
                       help="Enable subtask progress checking (default: False)")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                       help="Enable proprio image data recording (default: False)")
parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                       help="Output folder name under /robolab/output.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                       help="Verbose output (default: False)")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                       help="Debug output (default: False)")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                       help="Remote host for policy server (default: localhost)")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                       help="Remote port for policy server (default: 8000)")
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                       help="Full WebSocket URI for policy server.")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                       help="Number of actions to execute from each predicted chunk before requesting a new one.")
parser.add_argument("--dz-binarize-gripper", "--dz_binarize_gripper", action="store_true",
                    help="[DreamZero] Re-enable gripper binarization at 0.5 threshold")
parser.add_argument("--dz-resize", "--dz_resize", type=str, default="area", choices=["area", "linear", "pad"],
                    help="[DreamZero] Image resize method")
parser.add_argument("--remote-token", "--remote_token", type=str, default=None,
                    help="Bearer token for authenticated endpoints.")
parser.add_argument("--dz-cam2", "--dz_cam2", type=str, default="black",
                    choices=["black", "right", "head", "duplicate"],
                    help="[DreamZero] Second exterior camera source")
parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default",
                       help="Which instruction variant to use when a task defines multiple")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save (default: all)")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true",
                    help="Sample a random non-default background per task at registration time.")
parser.add_argument("--background-seed", "--background_seed", type=int, default=None,
                    help="Seed for reproducible per-task background sampling.")
# Dynamic prompting arguments
parser.add_argument("--check-every-n-steps", "--check_every_n_steps", type=int, default=15,
                    help="How often (in sim steps) to ask the VLM whether the current subtask is complete (default: 15)")
parser.add_argument("--topreward", action="store_true",
                    help="Enable live TOPReward tracking via Bedrock Qwen (default: False)", default=True)
parser.add_argument("--topreward-stride", "--topreward_stride", type=int, default=10,
                    help="Frame subsampling stride for TOPReward scoring (default: 10)")
parser.add_argument("--topreward-interval", "--topreward_interval", type=float, default=5.0,
                    help="Seconds between TOPReward scoring calls (default: 5.0)")

# parse the arguments
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
args_cli.save_videos = args_cli.video_mode != "none"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import logging
import re
from pathlib import Path

import torch
from tqdm import tqdm

from robolab.constants import PACKAGE_DIR, set_output_dir  # noqa
from robolab.core.environments.runtime import create_env  # noqa
from robolab.eval import create_client, summarize_run  # noqa
from robolab.core.environments.factory import get_envs  # noqa
from robolab.core.utils.print_utils import print_experiment_summary  # noqa
from robolab.core.logging.results import check_all_episodes_complete, check_run_complete  # noqa
from robolab.core.logging.results import init_experiment, summarize_experiment_results  # noqa
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams  # noqa
from robolab.core.world.world_state import get_world  # noqa
from robolab.core.utils.video_utils import VideoWriter  # noqa
from robolab.eval.episode import TimingStats  # noqa
import robolab.constants  # noqa

from robolab.harness.subtask_manager import get_next_subtask  # noqa
from robolab.harness.progress_monitor import ProgressMonitor  # noqa
from robolab.harness.memory_manager import MemoryManager  # noqa

# Update robolab.constants module settings from command line arguments
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

# Run automatic factory generation before main
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs  # noqa
if args_cli.policy == "dreamzero" and args_cli.dz_cam2 in ("right", "head"):
    from robolab.registrations.droid_jointpos.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa
    auto_register_droid_envs(
        task_dirs=args_cli.task_dirs, task=args_cli.task, cameras=WRIST_LEFT_RIGHT_HEAD,
        randomize_background=args_cli.randomize_background,
        background_seed=args_cli.background_seed,
    )
else:
    auto_register_droid_envs(
        task_dirs=args_cli.task_dirs, task=args_cli.task,
        randomize_background=args_cli.randomize_background,
        background_seed=args_cli.background_seed,
    )

logger = logging.getLogger(__name__)


def run_dynamic_prompting_episode(env, env_cfg, episode, client, *, headless=True, save_videos=True, video_mode="all", check_every_n_steps=15, reward_tracker=None, scene_output_dir=None):
    """Run a policy-controlled episode with reactive dynamic prompting.

    Instead of passing the same instruction every step, this function:
    1. Asks a VLM for the first subtask
    2. Feeds the current subtask to the policy at each step
    3. Every check_every_n_steps, asks the VLM if the subtask is complete
    4. On completion, updates memory and asks the VLM for the next subtask
    5. Stops when the VLM says the overall goal is achieved

    Returns:
        tuple: (harness_results, [], timing) — success determined by VLM, not sim state
    """
    timer = TimingStats()

    obs, _ = env.reset()
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    instruction = env_cfg.instruction
    action_dim = getattr(
        getattr(env, "action_manager", None),
        "total_action_dim",
        None,
    ) or env.action_space.shape[-1]

    clients = [client] * env.num_envs

    if env.recorder_manager is not None and hasattr(env.recorder_manager, 'set_hdf5_file'):
        env.recorder_manager.cfg.dataset_export_dir_path = robolab.constants.get_output_dir()
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r'[^\w\s]', '', instruction).replace(' ', '_')
    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_{episode}_env{env_id}" if env.num_envs > 1 else f"_{episode}"
            if save_sensor:
                sensor_dir = scene_output_dir or robolab.constants.get_output_dir()
                video_path = os.path.join(sensor_dir, f"{cleaned_instruction}{suffix}.mp4")
                video_writers_obs.append(VideoWriter(video_path, video_fps))
            if save_viewport:
                video_path_viewport = os.path.join(robolab.constants.get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(video_path_viewport, video_fps))

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    # --- Dynamic prompting state (per-env) ---
    harness_output_dir = Path(robolab.constants.get_output_dir())

    log_file = harness_output_dir / "harness.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [Harness] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    def hlog(msg, color="\033[96m"):
        print(f"{color}[Harness] {msg}\033[0m")
        logger.info(msg)

    # For now dynamic prompting runs on env 0 only; other envs get the raw instruction.
    # Multi-env harness support can be added later by tracking per-env memory/monitor.
    initial_frame = unpack_image_obs(obs, scale=0.5, env_id=0).get("combined_image")
    memory = MemoryManager(harness_output_dir / f"memory.md")
    memory.reset(instruction, initial_frame=initial_frame)
    monitor = ProgressMonitor(check_every_n_steps=check_every_n_steps)

    subtasks: dict[int, str] = {}
    plan_result = get_next_subtask(instruction, initial_frame, memory.get_memory())
    if plan_result.done:
        hlog(f"VLM says goal is already achieved: \"{instruction}\"")
    current_subtask = plan_result.subtask if not plan_result.done else instruction
    subtask_index = 0
    subtasks[subtask_index + 1] = current_subtask
    goal_achieved = plan_result.done
    hlog(f"Instruction: \"{instruction}\"")
    if not goal_achieved:
        hlog(f"Subtask {subtask_index + 1}: \"{current_subtask}\"")

    if reward_tracker is not None:
        reward_tracker.add_frame(initial_frame, step=0)
        reward_tracker.set_subtask_index(0, step=0)

    actual_steps = 0
    try:
        for step in tqdm(range(max_steps)):

            while not timeline.is_playing():
                kit_app.update()

            timer.start("policy_inference")
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            for env_id in env.active_env_ids:
                # env 0 gets the current subtask; others get the raw instruction
                instr = current_subtask if env_id == 0 else instruction
                ret = clients[env_id].infer(obs, instr, env_id=env_id)
                actions[env_id] = torch.tensor(ret["action"], device=env.device)
                if env_id == 0 or last_viz is None:
                    last_viz = ret.get("viz")
            timer.stop("policy_inference")

            if not headless and last_viz is not None:
                cv2.imshow(f"{instruction}", cv2.cvtColor(last_viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if robolab.constants.VISUALIZE:
                get_world(env).visualize()

            timer.start("env_step")
            obs, reward, term, trunc, info = env.step(actions)
            timer.stop("env_step")


            if save_videos:
                timer.start("video_write")
                for env_id in range(env.num_envs):
                    if env._frozen_envs[env_id]:
                        continue
                    if save_sensor:
                        frame_obs = unpack_image_obs(obs, scale=0.5, env_id=env_id).get("combined_image")
                        video_writers_obs[env_id].write(frame_obs)
                    if save_viewport:
                        frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                        video_writers_viewport[env_id].write(frame_vp)
                timer.stop("video_write")

            if reward_tracker is not None:
                rt_frame = unpack_image_obs(obs, scale=0.5, env_id=0).get("combined_image")
                reward_tracker.add_frame(rt_frame, step=step)

            actual_steps += 1

            # --- VLM completion check (env 0 only) ---
            if not goal_achieved and 0 in env.active_env_ids and step % check_every_n_steps == 0 and step > 0:
                frame = unpack_image_obs(obs, scale=0.5, env_id=0).get("combined_image")
                monitor.set_frame(frame)

                timer.start("vlm_check")
                mem_context = memory.context_for(current_subtask, subtask_index)
                result = monitor.check_completion(
                    current_subtask,
                    memory=mem_context,
                    before_frame=memory.last_frame(),
                )
                timer.stop("vlm_check")

                logger.info("Subtask result: completed=%s reason=%s", result["completed"], result.get("reason", ""))

                if result["completed"]:
                    hlog(f"Subtask done: {subtask_index + 1} \"{current_subtask}\" | {result['reason']}", color="\033[92m")
                    memory.update(frame, current_subtask)

                    timer.start("vlm_next_subtask")
                    plan_result = get_next_subtask(instruction, frame, memory.get_memory())
                    timer.stop("vlm_next_subtask")

                    if plan_result.done:
                        hlog(f"Goal achieved: \"{instruction}\"", color="\033[92m")
                        goal_achieved = True
                    else:
                        subtask_index += 1
                        current_subtask = plan_result.subtask
                        subtasks[subtask_index + 1] = current_subtask
                        if reward_tracker is not None:
                            reward_tracker.set_subtask_index(subtask_index, step=step)
                        hlog(f"Subtask {subtask_index + 1}: \"{current_subtask}\"")
                else:
                    hlog(f"Subtask not done: {subtask_index + 1} \"{current_subtask}\" | {result['reason']}", color="\033[93m")

            if goal_achieved:
                break
    finally:
        for vw in video_writers_obs + video_writers_viewport:
            try:
                vw.release()
            except Exception:
                logger.exception("Failed to release video writer")
        try:
            client.reset()
        except Exception:
            logger.exception("Failed to reset client after episode")
        if reward_tracker is not None:
            reward_tracker.stop()
            reward_tracker.save_plot(harness_output_dir / "topreward.png")
            reward_tracker.save_jsonl(harness_output_dir / "topreward.jsonl")
            reward_tracker.reset()
        import json
        with open(harness_output_dir / "subtasks.json", "w") as f:
            json.dump({"subtasks": subtasks}, f, indent=2)
        logger.removeHandler(file_handler)
        file_handler.close()

    harness_results = [{
        "env_id": 0,
        "success": goal_achieved,
        "step": actual_steps,
    }]
    timing = timer.to_dict(actual_steps)
    return harness_results, [], timing


def main():
    """Main function."""
    if args_cli.output_folder_name is None:
        args_cli.output_folder_name = get_timestamp() + f"_{args_cli.policy}_dynamic"
        if args_cli.instruction_type != "default":
            args_cli.output_folder_name += f"_{args_cli.instruction_type}"

    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.task:
        task_envs = get_envs(task=args_cli.task)
        filter_str = f"tasks: {', '.join(args_cli.task)}"
    elif args_cli.tag:
        task_envs = get_envs(tag=args_cli.tag)
        filter_str = f"tags: {', '.join(args_cli.tag)}"
    else:
        task_envs = get_envs()
        filter_str = "all"

    num_envs = args_cli.num_envs
    num_runs = args_cli.num_runs
    total_episodes = num_runs * num_envs

    print_experiment_summary(
        task_envs=task_envs,
        filter_str=filter_str,
        num_envs=num_envs,
        num_episodes=total_episodes,
        policy=args_cli.policy,
        instruction_type=args_cli.instruction_type,
        output_dir=output_dir,
    )

    episode_results_file, episode_results = init_experiment(output_dir)

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(episode_results=episode_results, env_name=task_env, num_episodes=total_episodes):
            print(f"\033[96m[RoboLab] Task `{task_env}` already done. Skipping.\033[0m")
            continue

        env, env_cfg = create_env(task_env,
            device=args_cli.device,
            num_envs=num_envs,
            use_fabric=True,
            instruction_type=args_cli.instruction_type,
            policy=args_cli.policy)

        client = create_client(
            args_cli.policy,
            remote_host=args_cli.remote_host,
            remote_port=args_cli.remote_port,
            remote_uri=args_cli.remote_uri,
            open_loop_horizon=args_cli.open_loop_horizon,
            api_token=args_cli.remote_token,
            binarize_gripper=args_cli.dz_binarize_gripper,
            resize=args_cli.dz_resize,
            cam2_source=args_cli.dz_cam2,
        )

        for run_idx in range(num_runs):

            run_episode_ids = [run_idx * num_envs + eid for eid in range(num_envs)]
            if all(check_run_complete(episode_results=episode_results, env_name=task_env, episode=ep_id) for ep_id in run_episode_ids):
                print(f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m")
                continue

            if args_cli.instruction_type != "default":
                run_name = task_env + f"_{args_cli.instruction_type}_{run_idx}"
            else:
                run_name = task_env + f"_{run_idx}"
            print(f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' (run {run_idx}, {num_envs} envs, dynamic prompting)\033[0m")

            ep_output_dir = os.path.join(scene_output_dir, "harness_logs", f"ep{run_idx}")
            os.makedirs(ep_output_dir, exist_ok=True)
            set_output_dir(ep_output_dir)

            reward_tracker = None
            if args_cli.topreward:
                from robolab.harness.live_topreward import LiveRewardTracker
                reward_tracker = LiveRewardTracker(
                    instruction=env_cfg.instruction,
                    stride=args_cli.topreward_stride,
                    score_interval=args_cli.topreward_interval,
                )
                reward_tracker.start()

            env_results, msgs, timing = run_dynamic_prompting_episode(
                env=env,
                env_cfg=env_cfg,
                episode=run_idx,
                client=client,
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
                num_envs=num_envs,
                run_idx=run_idx,
                run_name=run_name,
                task_env=task_env,
                scene_output_dir=ep_output_dir,
                policy=args_cli.policy,
                instruction_type=args_cli.instruction_type,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
                enable_subtask_progress=robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING,
            )

            env.reset_eval_state()

        env.close()

    summarize_experiment_results(episode_results, show_timing=True)

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
