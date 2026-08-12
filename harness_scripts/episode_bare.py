# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Barebones episode runner: policy inference + video recording only. No VLM checker."""

import os
import re

import cv2
import torch
from tqdm import tqdm

import robolab.constants
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.video_utils import VideoWriter
from robolab.eval.episode import TimingStats


def run_bare_episode(env, env_cfg, episode, client, *, headless=True, save_videos=True,
                     video_mode="all", scene_output_dir=None):
    """Run policy for max_steps, record video, no VLM termination."""
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
                viewport_dir = scene_output_dir or robolab.constants.get_output_dir()
                video_path_vp = os.path.join(viewport_dir, f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(video_path_vp, video_fps))

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    print(f"\033[96m[Bare] Running {max_steps} steps: \"{instruction}\"\033[0m")

    actual_steps = 0
    try:
        for step in tqdm(range(max_steps)):
            while not timeline.is_playing():
                kit_app.update()

            timer.start("policy_inference")
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            for env_id in env.active_env_ids:
                ret = client.infer(obs, instruction, env_id=env_id)
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

            actual_steps += 1
    finally:
        for vw in video_writers_obs + video_writers_viewport:
            try:
                vw.release()
            except Exception:
                pass
        try:
            client.reset()
        except Exception:
            pass

    harness_results = [{
        "env_id": 0,
        "success": False,
        "step": actual_steps,
    }]
    timing = timer.to_dict(actual_steps)
    return harness_results, [], timing
