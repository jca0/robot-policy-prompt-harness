# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Calibration episode runner — like dynamic prompting but scores each subtask
prompt with TOPReward delta and accumulates results into a CalibrationState."""

import json
import logging
import os
import re
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

import robolab.constants
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.video_utils import VideoWriter
from robolab.eval.episode import TimingStats
from robolab.harness.memory_manager import MemoryManager
from robolab.harness.progress_monitor import ProgressMonitor

from harness_scripts.prompt_calibration import (
    CalibrationState,
    ScoredPrompt,
    ScoredSequence,
    get_next_subtask_calibrated,
    reward_at_step,
)

logger = logging.getLogger(__name__)


def run_calibration_episode(
    env,
    env_cfg,
    episode: int,
    client,
    calibration_state: CalibrationState,
    *,
    headless: bool = True,
    save_videos: bool = True,
    video_mode: str = "all",
    check_every_n_steps: int = 15,
    subtask_timeout_steps: int = 150,
    reward_tracker=None,
    scene_output_dir=None,
):
    """Run one calibration episode.

    At each subtask boundary, records the TOPReward delta for that subtask prompt
    into calibration_state, then saves state to disk.

    Returns:
        tuple: (harness_results, [], timing)
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

    if env.recorder_manager is not None and hasattr(env.recorder_manager, "set_hdf5_file"):
        env.recorder_manager.cfg.dataset_export_dir_path = robolab.constants.get_output_dir()
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")
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
                viewport_dir = scene_output_dir or robolab.constants.get_output_dir()
                video_path_vp = os.path.join(viewport_dir, f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(video_path_vp, video_fps))

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    harness_output_dir = Path(robolab.constants.get_output_dir())
    log_file = harness_output_dir / "harness.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [Calibration] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    def hlog(msg, color="\033[96m"):
        print(f"{color}[Calibration] {msg}\033[0m")
        logger.info(msg)

    initial_frame = unpack_image_obs(obs, scale=0.5, env_id=0).get("combined_image")
    memory = MemoryManager(harness_output_dir / "memory.md")
    memory.reset(instruction, initial_frame=initial_frame)
    monitor = ProgressMonitor(check_every_n_steps=check_every_n_steps)

    subtask_log: list[dict] = []

    plan_result = get_next_subtask_calibrated(
        instruction, initial_frame, memory.get_memory(), calibration_state
    )
    if plan_result.done:
        hlog(f"VLM says goal is already achieved: \"{instruction}\"")
    current_subtask = plan_result.subtask if not plan_result.done else instruction
    subtask_index = 0
    subtask_start_step = 0
    goal_achieved = plan_result.done

    hlog(f"Instruction: \"{instruction}\"")
    if not goal_achieved:
        hlog(f"Subtask {subtask_index + 1}: \"{current_subtask}\" (temp={calibration_state.temperature:.2f})")

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
                instr = current_subtask if env_id == 0 else instruction
                ret = clients[env_id].infer(obs, instr, env_id=env_id)
                actions[env_id] = torch.tensor(ret["action"], device=env.device)
                if env_id == 0 or last_viz is None:
                    last_viz = ret.get("viz")
            timer.stop("policy_inference")

            if not headless and last_viz is not None:
                cv2.imshow(f"{instruction}", cv2.cvtColor(last_viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

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

            if not goal_achieved and 0 in env.active_env_ids and step % check_every_n_steps == 0 and step > 0:
                frame = unpack_image_obs(obs, scale=0.5, env_id=0).get("combined_image")
                monitor.set_frame(frame)
                subtask_elapsed = step - subtask_start_step

                timer.start("vlm_check")
                mem_context = memory.context_for(current_subtask, subtask_index)
                result = monitor.check_completion(
                    current_subtask,
                    memory=mem_context,
                    before_frame=memory.last_frame(),
                )
                timer.stop("vlm_check")

                subtask_done = result["completed"] or subtask_elapsed >= subtask_timeout_steps

                if subtask_done:
                    status = "succeeded" if result["completed"] else "timed_out"
                    hlog(
                        f"Subtask {status}: {subtask_index + 1} \"{current_subtask}\" | {result['reason']}",
                        color="\033[92m" if result["completed"] else "\033[91m",
                    )

                    # Only score prompts that succeeded
                    if result["completed"] and reward_tracker is not None:
                        snapshots = reward_tracker.snapshots
                        r_start = reward_at_step(snapshots, subtask_start_step)
                        r_end = reward_at_step(snapshots, step)
                        if r_start is not None and r_end is not None:
                            delta = r_end - r_start
                            scored = ScoredPrompt(
                                text=current_subtask,
                                reward_delta=delta,
                                episode=episode,
                            )
                            calibration_state.add(scored)
                            calibration_state.save()
                            hlog(f"Scored \"{current_subtask}\": delta={delta:+.3f}")

                    subtask_log.append({
                        "index": subtask_index + 1,
                        "subtask": current_subtask,
                        "status": status,
                        "steps_taken": subtask_elapsed,
                    })
                    if status == "timed_out":
                        memory.record_timeout(frame, current_subtask)
                    else:
                        memory.update(frame, current_subtask)

                    timer.start("vlm_next_subtask")
                    plan_result = get_next_subtask_calibrated(
                        instruction, frame, memory.get_memory(), calibration_state
                    )
                    timer.stop("vlm_next_subtask")

                    if plan_result.done:
                        hlog(f"Goal achieved: \"{instruction}\"", color="\033[92m")
                        goal_achieved = True
                    else:
                        subtask_index += 1
                        current_subtask = plan_result.subtask
                        subtask_start_step = step
                        if reward_tracker is not None:
                            reward_tracker.set_subtask_index(subtask_index, step=step)
                        hlog(f"Subtask {subtask_index + 1}: \"{current_subtask}\"")

                else:
                    hlog(
                        f"Subtask not done: {subtask_index + 1} \"{current_subtask}\" | "
                        f"{result['reason']} ({subtask_elapsed}/{subtask_timeout_steps} steps)",
                        color="\033[93m",
                    )

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
        if not goal_achieved:
            subtask_log.append({
                "index": subtask_index + 1,
                "subtask": current_subtask,
                "status": "abandoned",
                "steps_taken": actual_steps - subtask_start_step,
            })

        if reward_tracker is not None:
            reward_tracker.stop()
            reward_tracker.save_plot(harness_output_dir / "topreward.png")
            reward_tracker.save_jsonl(harness_output_dir / "topreward.jsonl")
            reward_tracker.save_subtask_plots(harness_output_dir, subtask_log)

            # Record the full subtask sequence with episode-level scoring
            snapshots = reward_tracker.snapshots
            r_start = reward_at_step(snapshots, 0)
            r_end = reward_at_step(snapshots, actual_steps)
            total_delta = (r_end - r_start) if (r_start is not None and r_end is not None) else 0.0
            sequence = ScoredSequence(
                subtasks=[entry["subtask"] for entry in subtask_log],
                goal_achieved=goal_achieved,
                total_reward_delta=total_delta,
                total_steps=actual_steps,
                episode=episode,
            )
            calibration_state.add_sequence(sequence)
            calibration_state.save()
            hlog(
                f"Recorded sequence ({'succeeded' if goal_achieved else 'failed'}): "
                f"{len(subtask_log)} subtasks, reward delta={total_delta:+.3f}"
            )

            reward_tracker.reset()

        with open(harness_output_dir / "subtasks.json", "w") as f:
            json.dump({
                "goal": instruction,
                "goal_achieved": goal_achieved,
                "subtasks": subtask_log,
            }, f, indent=2)

        logger.removeHandler(file_handler)
        file_handler.close()

    harness_results = [{
        "env_id": 0,
        "success": goal_achieved,
        "step": actual_steps,
    }]
    timing = timer.to_dict(actual_steps)
    return harness_results, [], timing
