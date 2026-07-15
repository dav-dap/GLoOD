from __future__ import annotations

from pathlib import Path
from itertools import cycle

import hydra
import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_FONT_SIZE = 16
PLOT_TITLE_SIZE = 18
PLOT_LEGEND_SIZE = 14


def _resolve_input_csvs(input_dir: Path) -> list[Path]:
    csv_paths = sorted(input_dir.glob("*/history.csv"))
    if not csv_paths:
        csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No history CSVs found in {input_dir!r}.")
    return csv_paths


def _load_csvs(paths: list[Path]) -> list[pd.DataFrame]:
    dataframes: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        try:
            df = df.apply(pd.to_numeric, errors="raise")
        except Exception as exc:
            raise ValueError(f"Non-numeric values found in {path}.") from exc
        if df.isna().any().any():
            raise ValueError(f"NaN values found in {path}.")
        dataframes.append(df)
    return dataframes


def _assert_same_shape(dfs: list[pd.DataFrame], paths: list[Path]) -> None:
    ref_shape = dfs[0].shape
    ref_cols = list(dfs[0].columns)
    for df, path in zip(dfs, paths):
        if df.shape != ref_shape:
            raise ValueError(f"{path} has shape {df.shape}, expected {ref_shape}.")
        if list(df.columns) != ref_cols:
            raise ValueError(f"{path} column mismatch versus reference file.")


def _compute_mean_std(dfs: list[pd.DataFrame], ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Compute plain mean and std across runs (axis=0), with no filtering."""
    arr = np.stack([df.to_numpy(dtype=float) for df in dfs], axis=0)  # shape: (runs, epochs, cols)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=ddof)
    return mean, std


def _compute_robust_center_bounds(
    dfs: list[pd.DataFrame],
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.stack([df.to_numpy(dtype=float) for df in dfs], axis=0)  # shape: (runs, epochs, cols)
    center = np.percentile(arr, 50.0, axis=0)
    lower = np.percentile(arr, lower_percentile, axis=0)
    upper = np.percentile(arr, upper_percentile, axis=0)
    return center, lower, upper


def _plot_with_band(
    ax: plt.Axes,
    x: np.ndarray,
    y_center: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    label: str,
    color: str | None = None,
    linestyle: str = "-",
    log_scale: bool = True,
    safe_log_bands: bool = True,
    log_eps: float = 1.0e-12,
) -> None:
    y_center = np.asarray(y_center, dtype=float)
    lower = np.asarray(y_lower, dtype=float)
    upper = np.asarray(y_upper, dtype=float)

    if log_scale and safe_log_bands:
        y_center = np.maximum(y_center, log_eps)
        # Keep all epochs; only floor plot values to avoid invalid log-scale bounds.
        lower = np.maximum(lower, log_eps)
        upper = np.maximum(upper, log_eps)

    ax.plot(x, y_center, linestyle=linestyle, label=label, linewidth=2, color=color)
    ax.fill_between(x, lower, upper, alpha=0.25, color=color)


def _resolve_epoch_axis(mean_df: pd.DataFrame) -> np.ndarray:
    if "epoch" in mean_df.columns and mean_df["epoch"].nunique() > 1:
        return mean_df["epoch"].to_numpy(dtype=float)
    return np.arange(len(mean_df), dtype=float)


def _display_metric_name(raw_name: str) -> str:
    """Map internal metric identifiers to clearer plot labels."""
    return raw_name.replace("cosine_similarity", "cosine_distance")


def _plot_loss(
    center_df: pd.DataFrame,
    lower_df: pd.DataFrame,
    upper_df: pd.DataFrame,
    epoch: np.ndarray,
    output_base: Path,
    band_label: str,
    safe_log_bands: bool,
    log_eps: float,
    y_min: float | None,
    y_max: float | None,
) -> bool:
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False

    if "train_loss" in center_df.columns:
        _plot_with_band(
            ax,
            epoch,
            center_df["train_loss"],
            lower_df["train_loss"],
            upper_df["train_loss"],
            "Train loss",
            safe_log_bands=safe_log_bands,
            log_eps=log_eps,
        )
        plotted = True

    if "validation_loss" in center_df.columns:
        _plot_with_band(
            ax,
            epoch,
            center_df["validation_loss"],
            lower_df["validation_loss"],
            upper_df["validation_loss"],
            "Validation loss",
            linestyle="--",
            safe_log_bands=safe_log_bands,
            log_eps=log_eps,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Epoch", fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel("Loss", fontsize=PLOT_FONT_SIZE)
    ax.set_title(f"Training vs validation loss ({band_label})", fontsize=PLOT_TITLE_SIZE)
    ax.set_yscale("log")
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
    ax.legend(fontsize=PLOT_LEGEND_SIZE)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_metrics(
    center_df: pd.DataFrame,
    lower_df: pd.DataFrame,
    upper_df: pd.DataFrame,
    epoch: np.ndarray,
    output_base: Path,
    band_label: str,
    safe_log_bands: bool,
    log_eps: float,
    y_min: float | None,
    y_max: float | None,
) -> bool:
    metric_columns = [
        col
        for col in center_df.columns
        if col not in {"epoch", "train_loss", "validation_loss"}
    ]
    if not metric_columns:
        return False

    base_names: list[str] = []
    for col in metric_columns:
        if col.startswith("train_"):
            base_names.append(col[len("train_"):])
        elif col.startswith("validation_"):
            base_names.append(col[len("validation_"):])
        else:
            base_names.append(col)

    unique_names = sorted(set(base_names))
    if not unique_names:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"]))
    plotted = False

    for name in unique_names:
        display_name = _display_metric_name(name)
        color = next(color_cycle)
        train_key = f"train_{name}"
        val_key = f"validation_{name}"

        if train_key in center_df.columns:
            _plot_with_band(
                ax,
                epoch,
                center_df[train_key],
                lower_df[train_key],
                upper_df[train_key],
                f"Train {display_name}",
                color=color,
                safe_log_bands=safe_log_bands,
                log_eps=log_eps,
            )
            plotted = True

        if val_key in center_df.columns:
            _plot_with_band(
                ax,
                epoch,
                center_df[val_key],
                lower_df[val_key],
                upper_df[val_key],
                f"Validation {display_name}",
                color=color,
                linestyle="--",
                safe_log_bands=safe_log_bands,
                log_eps=log_eps,
            )
            plotted = True

        if train_key not in center_df.columns and val_key not in center_df.columns and name in center_df.columns:
            _plot_with_band(
                ax,
                epoch,
                center_df[name],
                lower_df[name],
                upper_df[name],
                display_name,
                color=color,
                safe_log_bands=safe_log_bands,
                log_eps=log_eps,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Epoch", fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel("Metric value", fontsize=PLOT_FONT_SIZE)
    ax.set_title(f"Training vs validation metrics ({band_label})", fontsize=PLOT_TITLE_SIZE)
    ax.set_yscale("log")
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
    ax.legend(fontsize=PLOT_LEGEND_SIZE)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return True


@hydra.main(version_base=None, config_path="conf", config_name="process_stats_generic")
def main(cfg: DictConfig) -> None:
    with (env.out() / "conf_process_stats.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    input_dir = env.cwd() / cfg.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = env.out()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = _resolve_input_csvs(input_dir)
    dfs = _load_csvs(csv_paths)
    _assert_same_shape(dfs, csv_paths)

    band_mode = str(cfg.plot.band_mode).strip().lower()
    safe_log_bands = bool(cfg.plot.safe_log_bands)
    log_eps = float(cfg.plot.log_eps)
    robust_lower_percentile = float(cfg.plot.robust_lower_percentile)
    robust_upper_percentile = float(cfg.plot.robust_upper_percentile)
    loss_ymin = None if cfg.plot.loss_ymin is None else float(cfg.plot.loss_ymin)
    loss_ymax = None if cfg.plot.loss_ymax is None else float(cfg.plot.loss_ymax)
    metrics_ymin = None if cfg.plot.metrics_ymin is None else float(cfg.plot.metrics_ymin)
    metrics_ymax = None if cfg.plot.metrics_ymax is None else float(cfg.plot.metrics_ymax)

    if band_mode not in {"mean_std", "robust"}:
        raise ValueError("plot.band_mode must be one of: mean_std, robust.")
    if log_eps <= 0.0:
        raise ValueError("plot.log_eps must be > 0.")
    if not 0.0 <= robust_lower_percentile <= 100.0:
        raise ValueError("plot.robust_lower_percentile must be in [0, 100].")
    if not 0.0 <= robust_upper_percentile <= 100.0:
        raise ValueError("plot.robust_upper_percentile must be in [0, 100].")
    if robust_lower_percentile > robust_upper_percentile:
        raise ValueError("plot.robust_lower_percentile must be <= plot.robust_upper_percentile.")
    for name, y_min, y_max in (
        ("plot.loss_y", loss_ymin, loss_ymax),
        ("plot.metrics_y", metrics_ymin, metrics_ymax),
    ):
        if y_min is not None and y_min <= 0.0:
            raise ValueError(f"{name}min must be > 0 for log-scaled plots.")
        if y_max is not None and y_max <= 0.0:
            raise ValueError(f"{name}max must be > 0 for log-scaled plots.")
        if y_min is not None and y_max is not None and y_min >= y_max:
            raise ValueError(f"{name}min must be < {name}max.")

    mean, std = _compute_mean_std(dfs, ddof=1)
    mean_df = pd.DataFrame(mean, columns=dfs[0].columns)
    std_df = pd.DataFrame(std, columns=dfs[0].columns)

    if band_mode == "robust":
        center, lower, upper = _compute_robust_center_bounds(
            dfs,
            lower_percentile=robust_lower_percentile,
            upper_percentile=robust_upper_percentile,
        )
        band_label = (
            f"median with p{robust_lower_percentile:g}-p{robust_upper_percentile:g} band"
        )
    else:
        center = mean
        lower = mean - std
        upper = mean + std
        band_label = "mean +/- 1 sigma"

    center_df = pd.DataFrame(center, columns=dfs[0].columns)
    lower_df = pd.DataFrame(lower, columns=dfs[0].columns)
    upper_df = pd.DataFrame(upper, columns=dfs[0].columns)

    base = input_dir.name
    mean_csv = output_dir / f"{base}_mean.csv"
    std_csv = output_dir / f"{base}_std.csv"
    mean_df.to_csv(mean_csv, index=False)
    std_df.to_csv(std_csv, index=False)
    print(f"Averaged {len(dfs)} file(s) -> {mean_csv.name}")
    print(f"Sample std-devs         -> {std_csv.name}")
    if band_mode == "robust":
        lower_tag = f"p{robust_lower_percentile:g}".replace(".", "p")
        upper_tag = f"p{robust_upper_percentile:g}".replace(".", "p")
        median_csv = output_dir / f"{base}_median.csv"
        lower_csv = output_dir / f"{base}_{lower_tag}.csv"
        upper_csv = output_dir / f"{base}_{upper_tag}.csv"
        center_df.to_csv(median_csv, index=False)
        lower_df.to_csv(lower_csv, index=False)
        upper_df.to_csv(upper_csv, index=False)
        print(f"Robust center           -> {median_csv.name}")
        print(f"Robust lower band       -> {lower_csv.name}")
        print(f"Robust upper band       -> {upper_csv.name}")

    epoch = _resolve_epoch_axis(center_df)

    loss_base = output_dir / f"{base}_loss"
    metrics_base = output_dir / f"{base}_metrics"

    loss_plotted = _plot_loss(
        center_df,
        lower_df,
        upper_df,
        epoch,
        loss_base,
        band_label=band_label,
        safe_log_bands=safe_log_bands,
        log_eps=log_eps,
        y_min=loss_ymin,
        y_max=loss_ymax,
    )
    metrics_plotted = _plot_metrics(
        center_df,
        lower_df,
        upper_df,
        epoch,
        metrics_base,
        band_label=band_label,
        safe_log_bands=safe_log_bands,
        log_eps=log_eps,
        y_min=metrics_ymin,
        y_max=metrics_ymax,
    )

    if loss_plotted:
        print(f"Loss plots saved -> {loss_base.name}.pdf, {loss_base.name}.png")
    if metrics_plotted:
        print(f"Metrics plots saved -> {metrics_base.name}.pdf, {metrics_base.name}.png")
    if not loss_plotted and not metrics_plotted:
        print("Warning: no plottable loss/metrics columns were found.")


if __name__ == "__main__":
    main()
