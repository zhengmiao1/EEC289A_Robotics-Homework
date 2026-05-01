#!/usr/bin/env python3
"""Train the course Go2 locomotion policy with PPO.

Design goals
------------
1. Keep the file small and readable.
2. Preserve the original two-stage training behavior.
3. Make every output artifact explicit:
   - progress.json
   - summary.json
   - resolved_config.json
   - best_checkpoint/manifest.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    detect_gpu_name,
    ensure_environment_available,
    export_selected_checkpoint,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    resolve_latest_checkpoint_dir,
    save_json,
    set_runtime_env,
    stage_sequence,
    to_jsonable,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "run_default",
        help="Directory for checkpoints, logs, and summaries.",
    )
    parser.add_argument(
        "--stage",
        choices=["stage_1", "stage_2", "both"],
        default="both",
        help="Which training stage to run.",
    )
    parser.add_argument("--env-name", type=str, default=None, help="Optional environment name override.")
    parser.add_argument("--impl", choices=["jax", "warp"], default=None, help="Optional backend override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed override.")
    parser.add_argument(
        "--disable-domain-randomization",
        action="store_true",
        help="Disable domain randomization for debugging only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve the config and exit.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config before training.")

    # Small runtime overrides. These are useful in Colab and for debugging.
    parser.add_argument("--num-envs", type=int, default=None, help="Override the number of training environments.")
    parser.add_argument("--num-eval-envs", type=int, default=None, help="Override the number of evaluation environments.")
    parser.add_argument("--num-evals", type=int, default=None, help="Override the number of evaluation passes.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the PPO batch size.")
    parser.add_argument("--episode-length", type=int, default=None, help="Override the episode length.")
    parser.add_argument("--stage1-steps", type=int, default=None, help="Override stage 1 target environment steps.")
    parser.add_argument("--stage2-steps", type=int, default=None, help="Override stage 2 target environment steps.")
    parser.add_argument(
        "--policy-hidden-layer-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Override policy hidden sizes, for example: --policy-hidden-layer-sizes 256 256 128",
    )
    parser.add_argument(
        "--value-hidden-layer-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Override value hidden sizes, for example: --value-hidden-layer-sizes 256 256 128",
    )
    parser.add_argument("--num-minibatches", type=int, default=None, help="Override PPO num_minibatches.")
    parser.add_argument("--unroll-length", type=int, default=None, help="Override PPO unroll_length.")
    parser.add_argument("--num-updates-per-batch", type=int, default=None, help="Override PPO num_updates_per_batch.")

    parser.add_argument("--force-cpu", action="store_true", help="Force JAX onto CPU.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="go2-locomotion", help="W&B project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (default: output dir basename).")
    parser.add_argument(
        "--restore-checkpoint-dir",
        type=Path,
        default=None,
        help="Optional checkpoint directory to restore before the first selected stage.",
    )
    parser.add_argument(
        "--local-smoke",
        action="store_true",
        help="Use a tiny CPU-friendly training profile for local verification.",
    )
    return parser.parse_args()


def build_runtime_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    if args.local_smoke:
        overrides.update(
            {
                "force_cpu": True,
                "num_envs": 4,
                "num_eval_envs": 4,
                "num_evals": 1,
                "batch_size": 8,
                "episode_length": 200,
                "policy_hidden_layer_sizes": [64, 64],
                "value_hidden_layer_sizes": [64, 64],
                "num_minibatches": 2,
                "unroll_length": 10,
                "num_updates_per_batch": 2,
                "stage_1_num_timesteps": 4096,
                "stage_2_num_timesteps": 2048,
            }
        )

    if args.force_cpu:
        overrides["force_cpu"] = True
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    if args.num_eval_envs is not None:
        overrides["num_eval_envs"] = args.num_eval_envs
    if args.num_evals is not None:
        overrides["num_evals"] = args.num_evals
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.episode_length is not None:
        overrides["episode_length"] = args.episode_length
    if args.stage1_steps is not None:
        overrides["stage_1_num_timesteps"] = args.stage1_steps
    if args.stage2_steps is not None:
        overrides["stage_2_num_timesteps"] = args.stage2_steps
    if args.policy_hidden_layer_sizes is not None:
        overrides["policy_hidden_layer_sizes"] = list(args.policy_hidden_layer_sizes)
    if args.value_hidden_layer_sizes is not None:
        overrides["value_hidden_layer_sizes"] = list(args.value_hidden_layer_sizes)
    if args.num_minibatches is not None:
        overrides["num_minibatches"] = args.num_minibatches
    if args.unroll_length is not None:
        overrides["unroll_length"] = args.unroll_length
    if args.num_updates_per_batch is not None:
        overrides["num_updates_per_batch"] = args.num_updates_per_batch

    return overrides


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    if args.env_name:
        config["environment_name"] = args.env_name
    if args.impl:
        config["backend_impl"] = args.impl
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.disable_domain_randomization:
        config["use_domain_randomization"] = False

    config["runtime_overrides"] = build_runtime_overrides(args)
    if config["runtime_overrides"].get("force_cpu"):
        config["force_cpu"] = True
    return config


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def run_stage(
    *,
    stack: dict[str, Any],
    config: dict[str, Any],
    stage_name: str,
    output_dir: Path,
    restore_checkpoint_path: Path | None,
    wandb_run: Any = None,
) -> dict[str, Any]:
    """Train one curriculum stage and export the selected checkpoint."""
    ppo_networks = stack["ppo_networks"]
    ppo_train = stack["ppo_train"]
    registry = stack["registry"]
    wrapper = stack["wrapper"]
    locomotion_params = stack["locomotion_params"]

    env_name = config["environment_name"]
    impl = config["backend_impl"]

    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, impl)
    apply_stage_config(env_cfg, ppo_cfg, config, stage_name)

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))
    eval_env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))

    randomization_fn = None
    if config.get("use_domain_randomization", False):
        randomization_fn = registry.get_domain_randomizer(env_name)

    stage_dir = output_dir / stage_name
    checkpoint_root = stage_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    progress_history: list[dict[str, Any]] = []
    timing_points = [time.monotonic()]
    pbar: tqdm | None = None
    _prev_steps = [0]

    def progress_fn(num_steps: int, metrics: dict[str, Any]) -> None:
        nonlocal pbar
        timing_points.append(time.monotonic())
        record = {
            "num_steps": int(num_steps),
            "metrics": to_jsonable(metrics),
        }
        progress_history.append(record)
        save_json(stage_dir / "progress_live.json", progress_history)
        save_json(stage_dir / "latest_metrics.json", record)

        if wandb_run is not None:
            log_data = {f"{stage_name}/{k}": v for k, v in to_jsonable(metrics).items()}
            log_data["global_step"] = int(num_steps)
            log_data[f"{stage_name}/num_steps"] = int(num_steps)
            wandb_run.log(log_data)

        delta = num_steps - _prev_steps[0]
        _prev_steps[0] = num_steps
        postfix: dict[str, str] = {}
        reward = metrics.get("eval/episode_reward")
        if reward is not None:
            postfix["reward"] = f"{float(reward):.3f}"
        if pbar is not None:
            pbar.set_description(f"[{stage_name}]")
            pbar.set_postfix(postfix)
            pbar.update(delta)

    training_params = dict(ppo_cfg)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    num_eval_envs = int(_cfg_get(ppo_cfg, "num_eval_envs", config["training_defaults"]["num_eval_envs"]))
    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_cfg.network_factory,
    )

    save_json(
        stage_dir / "resolved_config.json",
        {
            "stage_name": stage_name,
            "env_name": env_name,
            "backend_impl": impl,
            "restore_checkpoint_path": str(restore_checkpoint_path) if restore_checkpoint_path else None,
            "resolved_env_config": to_jsonable(env_cfg),
            "resolved_ppo_config": to_jsonable(ppo_cfg),
        },
    )

    print(
        f"[{stage_name}] starting train: "
        f"env={env_name} impl={impl} "
        f"target_steps={int(ppo_cfg.num_timesteps)} "
        f"num_envs={int(ppo_cfg.num_envs)} "
        f"batch_size={int(ppo_cfg.batch_size)} "
        f"num_evals={int(ppo_cfg.num_evals)}",
        flush=True,
    )
    if restore_checkpoint_path is not None:
        print(f"[{stage_name}] restoring from checkpoint: {restore_checkpoint_path}", flush=True)

    pbar = tqdm(
        total=int(ppo_cfg.num_timesteps),
        unit="steps",
        desc=f"[{stage_name}] compiling",
        dynamic_ncols=True,
    )

    train_kwargs = dict(
        training_params,
        network_factory=network_factory,
        seed=int(config["seed"]),
        save_checkpoint_path=checkpoint_root,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        vision=False,
        restore_checkpoint_path=str(restore_checkpoint_path) if restore_checkpoint_path else None,
    )
    if randomization_fn is not None:
        train_kwargs["randomization_fn"] = randomization_fn

    train_fn = functools.partial(ppo_train, **train_kwargs)
    make_inference_fn, params, final_metrics = train_fn(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress_fn,
    )
    pbar.close()
    _ = make_inference_fn, params  # Explicitly show that training returned a policy.

    latest_checkpoint = resolve_latest_checkpoint_dir(checkpoint_root)
    best_export_dir = output_dir / "best_checkpoint"
    best_export_manifest = export_selected_checkpoint(stage_dir, best_export_dir)

    summary = {
        "stage_name": stage_name,
        "env_name": env_name,
        "backend_impl": impl,
        "num_progress_events": len(progress_history),
        "final_metrics": to_jsonable(final_metrics),
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "selected_checkpoint_source": best_export_manifest["source_checkpoint_dir"],
        "selected_checkpoint_export": best_export_manifest["exported_checkpoint_dir"],
        "selected_checkpoint_manifest": best_export_manifest,
        "compile_time_sec": (timing_points[1] - timing_points[0]) if len(timing_points) > 1 else None,
        "train_time_sec": (timing_points[-1] - timing_points[1]) if len(timing_points) > 1 else None,
    }

    save_json(stage_dir / "progress.json", progress_history)
    save_json(stage_dir / "summary.json", summary)
    print(
        f"[{stage_name}] finished: "
        f"latest_checkpoint={summary['latest_checkpoint']} "
        f"selected_checkpoint_source={summary['selected_checkpoint_source']}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    if args.print_config or args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        if args.dry_run:
            return

    selected_stages = stage_sequence(args.stage)

    if selected_stages == ["stage_2"]:
        restore_required = config["stage_2"].get("restore_previous_stage_checkpoint", False)
        if restore_required and args.restore_checkpoint_dir is None:
            raise SystemExit(
                "stage_2 is configured as a finetuning stage. "
                "Run '--stage both' or pass --restore-checkpoint-dir."
            )

    force_cpu = bool(config.get("force_cpu")) or bool(config.get("runtime_overrides", {}).get("force_cpu"))
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config_resolved.json", config)

    run_metadata = {
        "started_at_unix": time.time(),
        "gpu_name": detect_gpu_name(),
        "stages": selected_stages,
        "restore_checkpoint_dir": str(args.restore_checkpoint_dir.resolve()) if args.restore_checkpoint_dir else None,
        "force_cpu": force_cpu,
    }
    save_json(output_dir / "run_metadata.json", run_metadata)

    print(f"[run] output_dir={output_dir}", flush=True)
    print(f"[run] stages={selected_stages}", flush=True)
    if args.restore_checkpoint_dir:
        print(f"[run] initial restore checkpoint={args.restore_checkpoint_dir.resolve()}", flush=True)

    wandb_run = None
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("[wandb] wandb not installed, skipping. Run: pip install wandb", flush=True)
        else:
            run_name = args.wandb_run_name or output_dir.name
            wandb_run = _wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "stages": selected_stages,
                    "run_name": output_dir.name,
                    **{k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))},
                },
                resume="allow",
            )
            print(f"[wandb] run: {wandb_run.url}", flush=True)

    stack = lazy_import_stack()

    restore_checkpoint_path: Path | None = args.restore_checkpoint_dir.resolve() if args.restore_checkpoint_dir else None
    stage_summaries = []

    for stage_name in selected_stages:
        if stage_name == "stage_2" and config["stage_2"].get("restore_previous_stage_checkpoint", False):
            if restore_checkpoint_path is None:
                stage_1_summary_path = output_dir / "stage_1" / "summary.json"
                if stage_1_summary_path.exists():
                    stage_1_summary = load_json(stage_1_summary_path)
                    selected_source = stage_1_summary.get("selected_checkpoint_source")
                    if selected_source:
                        restore_checkpoint_path = Path(selected_source)
            if restore_checkpoint_path is None:
                stage_1_checkpoint_root = output_dir / "stage_1" / "checkpoints"
                restore_checkpoint_path = resolve_latest_checkpoint_dir(stage_1_checkpoint_root)
            if restore_checkpoint_path is None:
                raise RuntimeError("stage_2 requested finetuning, but no stage_1 checkpoint was found.")

        summary = run_stage(
            stack=stack,
            config=config,
            stage_name=stage_name,
            output_dir=output_dir,
            restore_checkpoint_path=restore_checkpoint_path,
            wandb_run=wandb_run,
        )
        stage_summaries.append(summary)

        selected_checkpoint_source = summary.get("selected_checkpoint_source")
        latest_checkpoint = summary.get("latest_checkpoint")
        restore_checkpoint_path = Path(selected_checkpoint_source) if selected_checkpoint_source else (
            Path(latest_checkpoint) if latest_checkpoint else None
        )

    run_metadata["finished_at_unix"] = time.time()
    run_metadata["wallclock_sec"] = run_metadata["finished_at_unix"] - run_metadata["started_at_unix"]
    run_metadata["stage_summaries"] = stage_summaries
    save_json(output_dir / "run_metadata.json", run_metadata)

    if wandb_run is not None:
        wandb_run.finish()

    print(json.dumps(run_metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
