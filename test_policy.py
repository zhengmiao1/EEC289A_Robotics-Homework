#!/usr/bin/env python3
"""Restore a trained checkpoint, render a deterministic demo, and score it.

This file is intentionally separated from train.py so that students can see the
difference between:
- training a policy
- restoring a policy
- rolling out a policy under a fixed command script
- evaluating a policy with benchmark metrics
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_specs import build_demo_segments, command_for_step, seconds_to_steps
from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from public_eval import clean_json_value, compute_metrics, compute_scores, normalize_rollout


ROOT = Path(__file__).resolve().parent

# These keys appear in Brax checkpoint metadata. We convert their string names
# back into callable initializers when restoring the network.
_KERNEL_INIT_FN_KEYS = (
    "policy_network_kernel_init_fn",
    "value_network_kernel_init_fn",
    "mean_kernel_init_fn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to a PPO checkpoint directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "demo_bundle",
        help="Directory for the video, rollout bundle, and eval JSON.",
    )
    parser.add_argument(
        "--stage-name",
        choices=["stage_1", "stage_2"],
        default="stage_2",
        help="Which stage config to use when building the eval environment.",
    )
    parser.add_argument(
        "--render-steps",
        type=int,
        default=0,
        help="Optional total control steps. Use 0 to derive the length from demo_rollout.segment_seconds.",
    )
    parser.add_argument("--render-width", type=int, default=960, help="Rendered video width.")
    parser.add_argument("--render-height", type=int, default=540, help="Rendered video height.")
    parser.add_argument("--render-camera", type=str, default="track", help="Camera name used for MuJoCo rendering.")
    parser.add_argument("--force-cpu", action="store_true", help="Force JAX onto CPU before restoring the checkpoint.")
    parser.add_argument("--episode-length", type=int, default=None, help="Optional override for environment episode length.")
    return parser.parse_args()


def load_policy_with_workaround(checkpoint_dir: Path, deterministic: bool) -> Any:
    """Restore a Brax PPO checkpoint saved by MuJoCo Playground / Brax training."""
    from brax.training import checkpoint as generic_checkpoint
    from brax.training import networks as training_networks
    from brax.training.agents.ppo import networks as ppo_networks
    from ml_collections import config_dict

    config_path = checkpoint_dir / "ppo_network_config.json"
    loaded_dict = json.loads(config_path.read_text())

    kwargs = loaded_dict["network_factory_kwargs"]
    if "activation" in kwargs and kwargs["activation"] is not None:
        kwargs["activation"] = training_networks.ACTIVATION[kwargs["activation"]]

    for init_key in _KERNEL_INIT_FN_KEYS:
        if init_key not in kwargs or kwargs[init_key] is None:
            continue
        kwargs[init_key] = training_networks.KERNEL_INITIALIZER[kwargs[init_key]]

    config = config_dict.create(**loaded_dict)
    network = generic_checkpoint.get_network(config, ppo_networks.make_ppo_networks)
    params = generic_checkpoint.load(checkpoint_dir)
    return ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)


def _force_command(state: Any, command: np.ndarray, jax: Any) -> Any:
    """Keep the command fixed so the rollout matches the scripted demo."""
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def _safe_float(value: Any) -> float:
    return float(np.asarray(value).item())


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["runtime_overrides"] = {}

    if args.episode_length is not None:
        config["runtime_overrides"]["episode_length"] = int(args.episode_length)
    if args.force_cpu:
        config["force_cpu"] = True
        config["runtime_overrides"]["force_cpu"] = True

    force_cpu = bool(config.get("force_cpu")) or bool(config.get("runtime_overrides", {}).get("force_cpu"))
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "demo_request.json",
        {
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "stage_name": args.stage_name,
            "render_steps": args.render_steps,
            "force_cpu": force_cpu,
        },
    )

    stack = lazy_import_stack()
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    jax = stack["jax"]
    media = stack["media"]

    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, args.stage_name)
    if args.episode_length is not None:
        env_cfg.episode_length = int(args.episode_length)

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))

    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)

    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    demo_segments = build_demo_segments(config)
    if args.render_steps > 0:
        total_steps = int(args.render_steps)
    else:
        segment_steps = seconds_to_steps(config["demo_rollout"]["segment_seconds"], env.dt)
        total_steps = int(segment_steps * len(demo_segments))
    if args.render_steps > 0:
        segment_steps = max(1, total_steps // len(demo_segments))

    rng = jax.random.PRNGKey(int(config["seed"]) + 123)
    state = reset_fn(rng)
    state = _force_command(state, np.asarray(demo_segments[0], dtype=np.float32), jax)
    trajectory = [state]

    command_xy = []
    measured_xy = []
    command_yaw = []
    measured_yaw = []
    joint_torques = []
    joint_velocities = []
    foot_slip_speed = []
    fell = []
    segment_ids = []
    base_xy_world = []

    done_step = None
    start_x = _safe_float(state.data.qpos[0])
    start_y = _safe_float(state.data.qpos[1])
    initial_xy = np.asarray([start_x, start_y], dtype=np.float32)

    for step_idx in range(total_steps):
        command = command_for_step(demo_segments, step_idx, total_steps)
        segment_idx = min(len(demo_segments) - 1, step_idx // segment_steps)
        state = _force_command(state, command, jax)

        rng, act_key = jax.random.split(rng)
        action, _ = policy(state.obs, act_key)
        state = step_fn(state, action)
        state = _force_command(state, command, jax)

        trajectory.append(state)

        command_xy.append(command[:2])
        measured_xy.append(np.asarray(env.get_local_linvel(state.data)[:2], dtype=np.float32))
        command_yaw.append(command[2])
        measured_yaw.append(np.asarray(env.get_gyro(state.data)[2], dtype=np.float32))
        joint_torques.append(np.asarray(state.data.actuator_force, dtype=np.float32))
        joint_velocities.append(np.asarray(state.data.qvel[6:], dtype=np.float32))
        feet_vel = np.asarray(state.data.sensordata[env._foot_linvel_sensor_adr], dtype=np.float32)
        foot_slip_speed.append(np.linalg.norm(feet_vel[:, :2], axis=-1).astype(np.float32))
        segment_ids.append(segment_idx)
        base_xy_world.append(np.asarray(state.data.qpos[:2], dtype=np.float32))

        state_done = bool(np.asarray(state.done))
        fell.append(state_done)
        if state_done and done_step is None:
            done_step = step_idx + 1
            break

    video_path = output_dir / "demo.mp4"
    frames = env.render(
        trajectory,
        height=int(args.render_height),
        width=int(args.render_width),
        camera=args.render_camera,
    )
    media.write_video(video_path, frames, fps=int(round(1.0 / env.dt)))

    rollout_npz = output_dir / "rollout_public_eval.npz"
    num_samples = len(command_xy)
    command_xy_arr = np.asarray(command_xy, dtype=np.float32)
    measured_xy_arr = np.asarray(measured_xy, dtype=np.float32)
    command_yaw_arr = np.asarray(command_yaw, dtype=np.float32)
    measured_yaw_arr = np.asarray(measured_yaw, dtype=np.float32)
    segment_ids_arr = np.asarray(segment_ids, dtype=np.int32)
    base_xy_world_arr = np.asarray(base_xy_world, dtype=np.float32)
    np.savez(
        rollout_npz,
        episode_id=np.zeros(num_samples, dtype=np.int32),
        command_lin_vel_xy=command_xy_arr,
        measured_lin_vel_xy=measured_xy_arr,
        command_yaw_rate=command_yaw_arr,
        measured_yaw_rate=measured_yaw_arr,
        fell=np.asarray(fell, dtype=bool),
        joint_torques=np.asarray(joint_torques, dtype=np.float32),
        joint_velocities=np.asarray(joint_velocities, dtype=np.float32),
        foot_slip_speed=np.asarray(foot_slip_speed, dtype=np.float32),
        segment_id=segment_ids_arr,
        base_xy_world=base_xy_world_arr,
    )

    per_segment_summary = []
    for segment_idx, segment_command in enumerate(demo_segments):
        idxs = np.nonzero(segment_ids_arr == segment_idx)[0]
        if idxs.size == 0:
            continue
        seg_start = initial_xy if idxs[0] == 0 else base_xy_world_arr[idxs[0] - 1]
        seg_end = base_xy_world_arr[idxs[-1]]
        per_segment_summary.append(
            {
                "segment_idx": int(segment_idx),
                "command": [float(x) for x in segment_command],
                "num_steps": int(idxs.size),
                "mean_measured_vx": float(np.mean(measured_xy_arr[idxs, 0])),
                "mean_measured_vy": float(np.mean(measured_xy_arr[idxs, 1])),
                "mean_measured_yaw": float(np.mean(measured_yaw_arr[idxs])),
                "mean_abs_vx_error": float(np.mean(np.abs(command_xy_arr[idxs, 0] - measured_xy_arr[idxs, 0]))),
                "mean_abs_vy_error": float(np.mean(np.abs(command_xy_arr[idxs, 1] - measured_xy_arr[idxs, 1]))),
                "mean_abs_yaw_error": float(np.mean(np.abs(command_yaw_arr[idxs] - measured_yaw_arr[idxs]))),
                "world_dx_m": float(seg_end[0] - seg_start[0]),
                "world_dy_m": float(seg_end[1] - seg_start[1]),
            }
        )

    final_state = trajectory[-1]
    rollout_summary = {
        "video_path": str(video_path),
        "rollout_npz": str(rollout_npz),
        "num_steps_simulated": num_samples,
        "terminated_early": done_step is not None,
        "done_step": done_step,
        "x_distance_m": _safe_float(final_state.data.qpos[0] - start_x),
        "y_offset_m": _safe_float(final_state.data.qpos[1] - start_y),
        "final_base_height_m": _safe_float(final_state.data.qpos[2]),
        "mean_command_vx": float(np.mean(command_xy_arr[:, 0])) if num_samples else 0.0,
        "mean_measured_vx": float(np.mean(measured_xy_arr[:, 0])) if num_samples else 0.0,
        "mean_command_vy": float(np.mean(command_xy_arr[:, 1])) if num_samples else 0.0,
        "mean_measured_vy": float(np.mean(measured_xy_arr[:, 1])) if num_samples else 0.0,
        "mean_abs_vx_error": float(np.mean(np.abs(command_xy_arr[:, 0] - measured_xy_arr[:, 0]))) if num_samples else 0.0,
        "mean_abs_vy_error": float(np.mean(np.abs(command_xy_arr[:, 1] - measured_xy_arr[:, 1]))) if num_samples else 0.0,
        "mean_abs_yaw_error": float(np.mean(np.abs(command_yaw_arr - measured_yaw_arr))) if num_samples else 0.0,
        "demo_segments": demo_segments,
        "segment_steps": int(segment_steps),
        "per_segment_summary": per_segment_summary,
    }
    save_json(output_dir / "demo_summary.json", rollout_summary)

    bundle = normalize_rollout(dict(np.load(rollout_npz)))
    metrics = compute_metrics(bundle)
    normalized_scores, composite_score = compute_scores(metrics, config["public_eval"]["metrics"])

    result = clean_json_value(
        {
            "homework_name": config["homework_name"],
            "robot": config["robot"],
            "environment_name": env_name,
            "stage_name": args.stage_name,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "num_steps": int(len(next(iter(bundle.values())))),
            "metrics": metrics,
            "normalized_scores": normalized_scores,
            "course_composite_score": composite_score,
            "rollout_summary": rollout_summary,
        }
    )
    save_json(output_dir / "public_eval.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
