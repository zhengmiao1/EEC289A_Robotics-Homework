#!/usr/bin/env python3
"""Publication-quality training curves from `train.py` progress JSON.

Reads ``<run_dir>/<stage>/progress.json`` (or ``progress_live.json``) written by
``train.py`` and produces vector figures suitable for Nature / Science–style
journals: single-column width, sans-serif typography, colorblind-safe palette,
minimal chart junk, optional uncertainty bands.

Example::

    python scripts/plot_train_curves.py \\
        --run-dir artifacts/run_v7 \\
        --metrics eval/episode_reward \\
        --out figures/go2_train

Dependencies: Python 3.10+, numpy, matplotlib (no JAX / Brax required).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _is_scalar_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(float(v))
    if isinstance(v, list) and len(v) == 1:
        return _is_scalar_number(v[0])
    return False


def _to_float(v: Any) -> float:
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    return float(v)


def load_progress_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def history_to_arrays(
    history: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return steps (int64) and aligned float arrays for each numeric metric key."""
    all_keys: set[str] = set()
    for row in history:
        m = row.get("metrics")
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if _is_scalar_number(v):
                all_keys.add(k)

    ordered_keys = sorted(all_keys)
    steps: list[int] = []
    buckets: dict[str, list[float]] = {k: [] for k in ordered_keys}

    for row in history:
        if "num_steps" not in row or "metrics" not in row:
            continue
        steps.append(int(row["num_steps"]))
        m = row["metrics"]
        if not isinstance(m, dict):
            for k in ordered_keys:
                buckets[k].append(float("nan"))
            continue
        for k in ordered_keys:
            v = m.get(k)
            if v is not None and _is_scalar_number(v):
                buckets[k].append(_to_float(v))
            else:
                buckets[k].append(float("nan"))

    if not steps:
        return np.array([], dtype=np.int64), {}

    arrays = {k: np.asarray(buckets[k], dtype=np.float64) for k in ordered_keys}
    return np.asarray(steps, dtype=np.int64), arrays


def find_progress_file(stage_dir: Path) -> Path:
    for name in ("progress.json", "progress_live.json"):
        p = stage_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No progress.json or progress_live.json under {stage_dir}")


def ema_smooth(y: np.ndarray, span: float) -> np.ndarray:
    """Exponential moving average; NaNs preserved. span ~ number of steps for decay."""
    if span <= 0 or y.size == 0:
        return y
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(y, dtype=np.float64)
    prev = np.nan
    for i, val in enumerate(y):
        if np.isnan(val):
            out[i] = np.nan
            continue
        if np.isnan(prev):
            out[i] = val
        else:
            out[i] = alpha * val + (1.0 - alpha) * prev
        prev = out[i]
    return out


# Okabe–Ito (colorblind-safe); first is blue, good for primary curve
OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def configure_matplotlib_publication(
    *,
    width_mm: float,
    height_mm: float,
    font_size_pt: float,
    dpi: float,
) -> None:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    width_in = width_mm / 25.4
    height_in = height_mm / 25.4

    plt.rcParams.update(
        {
            "figure.figsize": (width_in, height_in),
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": font_size_pt,
            "axes.titlesize": font_size_pt,
            "axes.labelsize": font_size_pt,
            "xtick.labelsize": font_size_pt - 0.5,
            "ytick.labelsize": font_size_pt - 0.5,
            "legend.fontsize": font_size_pt - 1,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#000000",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.borderpad": 0.2,
            "legend.handlelength": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )
    mpl.rcParams["axes.unicode_minus"] = False


def _safe_metric_pair(metric: str) -> tuple[str, str | None]:
    """Return (value_key, std_key) if a plausible std companion exists."""
    if metric.endswith("_std"):
        return metric, None
    std_key = f"{metric}_std"
    return metric, std_key


def plot_runs(
    series_list: list[tuple[str, str, np.ndarray, dict[str, np.ndarray]]],
    *,
    metrics: list[str],
    out_base: Path,
    ema_span: float,
    formats: Iterable[str],
    dpi: float,
    width_mm: float,
    height_per_row_mm: float,
    font_size_pt: float,
    x_scale: str,
    y_labels: dict[str, str] | None,
) -> None:
    import matplotlib.pyplot as plt

    nrows = len(metrics)
    height_mm = max(height_per_row_mm * nrows, height_per_row_mm)
    configure_matplotlib_publication(
        width_mm=width_mm,
        height_mm=height_mm,
        font_size_pt=font_size_pt,
        dpi=dpi,
    )

    fig, axes = plt.subplots(nrows, 1, sharex=True, squeeze=False)
    axes_flat = axes.ravel()
    use_letters = nrows >= 2

    for mi, metric in enumerate(metrics):
        ax = axes_flat[mi]
        if use_letters:
            letter = chr(ord("a") + mi)
            ax.text(
                -0.22,
                1.02,
                letter,
                transform=ax.transAxes,
                fontsize=font_size_pt + 1,
                fontweight="bold",
                va="bottom",
                ha="right",
            )
        value_key, std_key_template = _safe_metric_pair(metric)
        std_key = std_key_template if std_key_template else None

        for si, (label, _stage, steps, arrs) in enumerate(series_list):
            color = OKABE_ITO[si % len(OKABE_ITO)]
            if value_key not in arrs:
                continue
            y = np.asarray(arrs[value_key], dtype=np.float64)
            y_plot = ema_smooth(y, ema_span) if ema_span > 0 else y
            x = steps.astype(np.float64)
            if x_scale == "million":
                x = x / 1e6
            elif x_scale == "billion":
                x = x / 1e9

            ax.plot(x, y_plot, color=color, linewidth=1.0, label=label, clip_on=False)

            std_k = f"{value_key}_std" if std_key is None else std_key
            if std_k in arrs:
                err = np.asarray(arrs[std_k], dtype=np.float64)
                if np.any(np.isfinite(err)):
                    lo = y_plot - err
                    hi = y_plot + err
                    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)

        ylab = (y_labels or {}).get(metric)
        if not ylab:
            ylab = _pretty_metric_label(metric)
        ax.set_ylabel(ylab)
        ax.minorticks_off()

    xlab = "Environment steps"
    if x_scale == "million":
        xlab = "Environment steps (×10⁶)"
    elif x_scale == "billion":
        xlab = "Environment steps (×10⁹)"
    axes_flat[-1].set_xlabel(xlab)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(loc="best", ncol=1)

    fig.align_ylabels(axes_flat)
    left_pad = 0.20 if (nrows >= 2) else 0.14
    fig.subplots_adjust(left=left_pad, right=0.98, top=0.97, bottom=0.16, hspace=0.08)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        dest = out_base.with_suffix(f".{fmt}")
        fig.savefig(dest, bbox_inches="tight", pad_inches=0.02, transparent=False)
    plt.close(fig)


def _pretty_metric_label(metric: str) -> str:
    s = metric.replace("eval/", "Eval. ").replace("training/", "Train. ")
    s = s.replace("/", " — ").replace("_", " ")
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Training output directory (e.g. artifacts/run_v7). May be repeated to overlay runs.",
    )
    p.add_argument(
        "--run-label",
        action="append",
        default=[],
        help="Legend label for each --run-dir (same order). Default: directory name.",
    )
    p.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Stage subdirectory (e.g. stage_1). Default: stage_1 and stage_2 if present.",
    )
    p.add_argument(
        "--metrics",
        default="eval/episode_reward",
        help="Comma-separated metric keys inside progress records (e.g. eval/episode_reward,eval/episode_length).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("figures/train_curves"),
        help="Output path without suffix; .pdf and .png written unless --formats overrides.",
    )
    p.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated: pdf, png, svg.",
    )
    p.add_argument("--dpi", type=float, default=300.0, help="Raster export DPI (PNG).")
    p.add_argument("--width-mm", type=float, default=89.0, help="Figure width (Nature single column ≈ 89 mm).")
    p.add_argument("--row-height-mm", type=float, default=42.0, help="Height per metric row.")
    p.add_argument("--font-size-pt", type=float, default=8.0, help="Base font size (Nature often 7–8 pt).")
    p.add_argument(
        "--x-scale",
        choices=("1", "million", "billion"),
        default="million",
        help="Divide environment steps for axis readability.",
    )
    p.add_argument(
        "--ema-span",
        type=float,
        default=0.0,
        help="If > 0, exponential moving average smoothing (typical 3–11). 0 disables.",
    )
    return p.parse_args()


def default_stages(run_dir: Path) -> list[str]:
    found: list[str] = []
    for name in ("stage_1", "stage_2"):
        if (run_dir / name).is_dir():
            found.append(name)
    if found:
        return found
    raise FileNotFoundError(f"No stage_1 or stage_2 under {run_dir}")


def main() -> None:
    args = parse_args()
    if not args.run_dir:
        raise SystemExit("Provide at least one --run-dir.")

    run_dirs = [d.resolve() for d in args.run_dir]
    labels = list(args.run_label)
    while len(labels) < len(run_dirs):
        labels.append(run_dirs[len(labels)].name)

    stages = list(args.stage) if args.stage else None
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    series_list: list[tuple[str, str, np.ndarray, dict[str, np.ndarray]]] = []

    for run_dir, label in zip(run_dirs, labels):
        stage_names = stages or default_stages(run_dir)
        for st in stage_names:
            prog_path = find_progress_file(run_dir / st)
            hist = load_progress_json(prog_path)
            steps, arrs = history_to_arrays(hist)
            if steps.size == 0:
                print(f"[warn] no data in {prog_path}", flush=True)
                continue
            leg = f"{label} ({st})" if len(stage_names) > 1 else label
            series_list.append((leg, st, steps, arrs))

    if not series_list:
        raise SystemExit("No progress data found; check --run-dir and stages.")

    plot_runs(
        series_list,
        metrics=metrics,
        out_base=args.out.resolve(),
        ema_span=args.ema_span,
        formats=formats,
        dpi=args.dpi,
        width_mm=args.width_mm,
        height_per_row_mm=args.row_height_mm,
        font_size_pt=args.font_size_pt,
        x_scale=args.x_scale,
        y_labels=None,
    )

    for fmt in formats:
        dest = args.out.resolve().with_suffix(f".{fmt}")
        print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
