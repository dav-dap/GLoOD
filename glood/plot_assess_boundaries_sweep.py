from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_run_csvs(input_dir: Path, csv_name: str) -> list[tuple[int, Path]]:
    paths: list[tuple[int, Path]] = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            run_id = int(child.name)
        except ValueError:
            continue
        csv_path = child / csv_name
        if csv_path.is_file():
            paths.append((run_id, csv_path))
    if not paths:
        raise FileNotFoundError(f"No run folders with '{csv_name}' found under {input_dir}.")
    return paths


def _load_and_validate(paths: list[tuple[int, Path]]) -> pd.DataFrame:
    required_cols = {
        "pair",
        "epsilon",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "fpr",
        "fnr",
        "iou",
        "dice",
    }
    frames: list[pd.DataFrame] = []
    for run_id, path in paths:
        df = pd.read_csv(path)
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        df = df.copy()
        df["run_id"] = run_id
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["epsilon"] = pd.to_numeric(merged["epsilon"], errors="raise")
    metric_cols = sorted(required_cols.difference({"pair", "epsilon"}))
    for col in metric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="raise")
    merged = merged.sort_values(["epsilon", "run_id", "pair"]).reset_index(drop=True)
    return merged


def _summarize_by_epsilon_pair(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg_cols = [*metrics]
    if "n_rows" in df.columns:
        agg_cols = ["n_rows", *agg_cols]

    grouped = df.groupby(["epsilon", "pair"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        col
        if isinstance(col, str)
        else col[0]
        if col[1] == ""
        else f"{col[0]}_{col[1]}"
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.sort_values(["epsilon", "pair"]).reset_index(drop=True)
    std_cols = [col for col in grouped.columns if col.endswith("_std")]
    for col in std_cols:
        grouped[col] = grouped[col].fillna(0.0)
    return grouped


def _plot_metric_vs_epsilon(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    pairs: list[str],
    output_dir: Path,
    x_log_scale: bool,
    y_log_scale: bool,
) -> bool:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary_df.columns or std_col not in summary_df.columns:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for pair in pairs:
        sub = summary_df[summary_df["pair"] == pair].sort_values("epsilon")
        if sub.empty:
            continue

        x = sub["epsilon"].to_numpy(dtype=float)
        y = sub[mean_col].to_numpy(dtype=float)
        y_std = sub[std_col].to_numpy(dtype=float)
        lower = y - y_std
        upper = y + y_std

        if y_log_scale:
            eps = 1.0e-12
            y = np.clip(y, eps, None)
            lower = np.clip(lower, eps, None)
            upper = np.clip(upper, eps, None)

        ax.plot(x, y, marker="o", linewidth=2, label=pair)
        ax.fill_between(x, lower, upper, alpha=0.2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("epsilon")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs epsilon")
    if x_log_scale:
        ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    base = output_dir / f"epsilon_sweep_{metric}"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_metric_grid(
    summary_df: pd.DataFrame,
    *,
    metrics: list[str],
    pairs: list[str],
    output_dir: Path,
    x_log_scale: bool,
    y_log_scale: bool,
    ncols: int,
) -> bool:
    if not metrics:
        return False

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()

    any_plotted = False
    for ax, metric in zip(axes_arr, metrics):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in summary_df.columns or std_col not in summary_df.columns:
            ax.set_visible(False)
            continue

        metric_plotted = False
        for pair in pairs:
            sub = summary_df[summary_df["pair"] == pair].sort_values("epsilon")
            if sub.empty:
                continue

            x = sub["epsilon"].to_numpy(dtype=float)
            y = sub[mean_col].to_numpy(dtype=float)
            y_std = sub[std_col].to_numpy(dtype=float)
            lower = y - y_std
            upper = y + y_std

            if y_log_scale:
                eps = 1.0e-12
                y = np.clip(y, eps, None)
                lower = np.clip(lower, eps, None)
                upper = np.clip(upper, eps, None)

            ax.plot(x, y, marker="o", linewidth=2, label=pair)
            ax.fill_between(x, lower, upper, alpha=0.2)
            metric_plotted = True
            any_plotted = True

        if not metric_plotted:
            ax.set_visible(False)
            continue

        ax.set_title(metric)
        ax.set_xlabel("epsilon")
        ax.set_ylabel(metric)
        if x_log_scale:
            ax.set_xscale("log")
        if y_log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    for ax in axes_arr[len(metrics):]:
        ax.set_visible(False)

    if not any_plotted:
        plt.close(fig)
        return False

    handles, labels = [], []
    for ax in axes_arr:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    base = output_dir / "epsilon_sweep_all_metrics"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


@hydra.main(version_base=None, config_path="conf", config_name="plot_assess_boundaries_sweep_generic")
def main(cfg: DictConfig) -> None:
    with (env.out() / "conf_plot_assess_boundaries_sweep.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    input_dir = env.cwd() / cfg.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = env.out()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / str(cfg.outputs.csv_subdir)
    plot_dir = output_dir / str(cfg.outputs.plot_subdir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_name = str(cfg.sweep.input_csv_name)
    run_csvs = _resolve_run_csvs(input_dir, csv_name)
    merged_df = _load_and_validate(run_csvs)

    requested_pairs = [str(x) for x in cfg.sweep.pairs]
    if requested_pairs:
        merged_df = merged_df[merged_df["pair"].isin(requested_pairs)].copy()
    if merged_df.empty:
        raise ValueError("No rows left after filtering requested pairs.")
    if not requested_pairs:
        requested_pairs = sorted(merged_df["pair"].unique().tolist())

    metric_list = [str(x) for x in cfg.sweep.metrics]
    summary_df = _summarize_by_epsilon_pair(merged_df, metric_list)

    merged_path = csv_dir / str(cfg.outputs.merged_csv)
    summary_path = csv_dir / str(cfg.outputs.summary_csv)
    merged_df.to_csv(merged_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plotted_metrics: list[str] = []
    for metric in metric_list:
        plotted = _plot_metric_vs_epsilon(
            summary_df,
            metric=metric,
            pairs=requested_pairs,
            output_dir=plot_dir,
            x_log_scale=bool(cfg.plotting.x_log_scale),
            y_log_scale=bool(cfg.plotting.y_log_scale),
        )
        if plotted:
            plotted_metrics.append(metric)

    grid_plotted = _plot_metric_grid(
        summary_df,
        metrics=metric_list,
        pairs=requested_pairs,
        output_dir=output_dir,
        x_log_scale=bool(cfg.plotting.x_log_scale),
        y_log_scale=bool(cfg.plotting.y_log_scale),
        ncols=int(cfg.plotting.grid_ncols),
    )

    print(f"Loaded runs: {len(run_csvs)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged CSV -> {merged_path}")
    print(f"Summary CSV -> {summary_path}")
    if plotted_metrics:
        print(f"Plotted metrics: {', '.join(plotted_metrics)}")
    else:
        print("No metric plots were generated.")
    if grid_plotted:
        print(f"Combined subplot figure -> {output_dir / 'epsilon_sweep_all_metrics.png'}")


if __name__ == "__main__":
    main()
