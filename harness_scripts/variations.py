# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Environment registration, job building, and runtime variation helpers."""

import re
from dataclasses import dataclass, field
from itertools import product

from robolab.core.environments.factory import get_envs, get_envs_by_tag


@dataclass
class EvalJob:
    base_task_env: str
    display_name: str
    task_name: str
    table_material: str | None = None
    camera_event: dict | None = None
    extra_fields: dict = field(default_factory=dict)


def register_envs(args_cli) -> None:
    """Dispatch environment registration based on CLI variation flags."""
    from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs

    has_bg = args_cli.backgrounds is not None
    has_lighting = (args_cli.lighting_intensities is not None or
                    args_cli.lighting_types is not None)

    if has_bg:
        from robolab.registrations.droid_jointpos.auto_env_registrations_bg_variations import (
            auto_register_droid_envs_bg_variations,
        )
        auto_register_droid_envs_bg_variations(
            backgrounds=args_cli.backgrounds,
            tasks=args_cli.task,
        )

    if has_lighting:
        from robolab.registrations.droid_jointpos.auto_env_registrations_lighting_variations import (
            auto_register_droid_envs_light_intensity,
            auto_register_droid_envs_shadows,
            auto_register_droid_envs_colored_lights,
        )
        if args_cli.lighting_intensities:
            for intensity in args_cli.lighting_intensities:
                auto_register_droid_envs_light_intensity(lighting_intensity=intensity)
        if args_cli.lighting_types:
            if "directional" in args_cli.lighting_types:
                auto_register_droid_envs_shadows()
            color_types = [t for t in args_cli.lighting_types if t in ("red", "blue", "green")]
            if color_types:
                auto_register_droid_envs_colored_lights()

    if not has_bg and not has_lighting:
        auto_register_droid_envs(
            task=args_cli.task,
            randomize_background=args_cli.randomize_background,
            background_seed=args_cli.background_seed,
        )

    if (has_bg or has_lighting) and (args_cli.table_materials or args_cli.camera_variations):
        auto_register_droid_envs(
            task=args_cli.task,
            randomize_background=args_cli.randomize_background,
            background_seed=args_cli.background_seed,
        )


def build_eval_jobs(args_cli) -> list[EvalJob]:
    """Build the Cartesian product of (envs x table_materials x camera_variations)."""
    has_bg = args_cli.backgrounds is not None
    has_lighting = (args_cli.lighting_intensities is not None or
                    args_cli.lighting_types is not None)

    # Collect base envs
    if has_bg:
        base_envs = get_envs_by_tag("background_variations")
    elif has_lighting:
        envs = []
        if args_cli.task:
            for task in args_cli.task:
                if args_cli.lighting_intensities:
                    envs.extend([f"{task}_LightingIntensity{i}" for i in args_cli.lighting_intensities])
                if args_cli.lighting_types:
                    if "directional" in args_cli.lighting_types:
                        envs.append(f"{task}_Directional")
                    for color in ["red", "blue", "green"]:
                        if color in args_cli.lighting_types:
                            envs.append(f"{task}_{color.capitalize()}Light")
        base_envs = envs
    else:
        base_envs = get_envs(task=args_cli.task) if args_cli.task else get_envs()

    materials = [(m, m) for m in args_cli.table_materials] if args_cli.table_materials else [(None, None)]

    if args_cli.camera_variations:
        cam_events = _build_camera_events(args_cli.camera_variations)
    else:
        cam_events = [(None, None)]

    jobs = []
    for env_name, (mat_label, mat), (cam_label, cam_event) in product(base_envs, materials, cam_events):
        task_name = _extract_task_name(env_name)

        parts = [env_name]
        if mat_label:
            parts.append(mat_label)
        if cam_label:
            parts.append(cam_label)
        display_name = "_".join(parts)

        extra = {}
        if has_bg and "_bg_" in env_name:
            extra["background"] = env_name.split("_bg_")[-1]
        if has_lighting:
            extra.update(_extract_lighting_params(env_name))
        if mat:
            extra["table_material"] = mat
        if cam_label:
            extra["camera_variation"] = cam_label

        jobs.append(EvalJob(
            base_task_env=env_name,
            display_name=display_name,
            task_name=task_name,
            table_material=mat,
            camera_event=cam_event,
            extra_fields=extra,
        ))

    return jobs


def change_table_material(material_name: str) -> bool:
    """Change the table top material at runtime by modifying the USD material binding."""
    import omni.usd
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("Warning: No stage available")
        return False

    table_top_prim = None
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath()).lower()
        prim_name = prim.GetName().lower()
        if prim_name == "top" and "table" in prim_path and "franka" not in prim_path:
            table_top_prim = prim
            break

    if table_top_prim is None:
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath()).lower()
            if "table" in prim_path and "franka" not in prim_path and prim.GetTypeName() in ["Cube", "Mesh"]:
                table_top_prim = prim
                break

    if table_top_prim is None:
        print("Warning: No table mesh found to change material")
        return False

    material_prim = None
    for pattern in [f"/Root/Looks/{material_name}", f"/World/Looks/{material_name}", f"/world/Looks/{material_name}"]:
        potential = stage.GetPrimAtPath(pattern)
        if potential.IsValid():
            material_prim = potential
            break

    if material_prim is None:
        for prim in stage.Traverse():
            if prim.GetName() == material_name and prim.GetTypeName() == "Material":
                material_prim = prim
                break

    if material_prim is None:
        print(f"Warning: Material '{material_name}' not found in stage")
        return False

    material = UsdShade.Material(material_prim)
    binding_api = UsdShade.MaterialBindingAPI(table_top_prim)
    binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants)
    print(f"Changed table material to: {material_name} ({material_prim.GetPath()})")
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_task_name(env_name: str) -> str:
    for suffix in ["_bg_", "_LightingIntensity", "_Directional", "_RedLight", "_BlueLight", "_GreenLight"]:
        if suffix in env_name:
            return env_name.split(suffix)[0]
    return env_name


def _build_camera_events(presets: list[str]) -> list[tuple[str, dict]]:
    from robolab.core.events.reset_camera import reset_camera_pose_uniform
    from isaaclab.managers import EventTermCfg as EventTerm

    EXT_NAMES = ["over_shoulder_left_camera"]
    EXT_RANGE = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1),
                 "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)}
    WRIST_NAMES = ["wrist_cam"]
    WRIST_RANGE = {"x": (-0.015, 0.015), "y": (-0.015, 0.015),
                   "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)}

    results = []
    for preset in presets:
        if preset == "external":
            results.append(("cam_external", {
                "randomize_external_camera": EventTerm(
                    func=reset_camera_pose_uniform, mode="reset",
                    params={"camera_names": EXT_NAMES, "pose_range": EXT_RANGE})}))
        elif preset == "wrist":
            results.append(("cam_wrist", {
                "randomize_wrist_cam": EventTerm(
                    func=reset_camera_pose_uniform, mode="reset",
                    params={"camera_names": WRIST_NAMES, "pose_range": WRIST_RANGE})}))
        elif preset == "both":
            results.append(("cam_both", {
                "randomize_external_camera": EventTerm(
                    func=reset_camera_pose_uniform, mode="reset",
                    params={"camera_names": EXT_NAMES, "pose_range": EXT_RANGE}),
                "randomize_wrist_cam": EventTerm(
                    func=reset_camera_pose_uniform, mode="reset",
                    params={"camera_names": WRIST_NAMES, "pose_range": WRIST_RANGE})}))
    return results


def _extract_lighting_params(env_name: str) -> dict:
    m = re.search(r'_LightingIntensity(\d+)', env_name)
    if m:
        return {"lighting_intensity": int(m.group(1)), "lighting_color": "natural", "lighting_type": "domelight"}
    if "_Directional" in env_name:
        return {"lighting_intensity": 200, "lighting_color": "natural", "lighting_type": "directional"}
    for suffix, color in [("_RedLight", "red"), ("_BlueLight", "blue"), ("_GreenLight", "green")]:
        if suffix in env_name:
            return {"lighting_intensity": 100, "lighting_color": color, "lighting_type": "sphere"}
    return {}
