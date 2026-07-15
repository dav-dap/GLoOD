from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env
from glood.src.assessment.force_metrics import compute_series_force_components
from glood.src.assessment.mask_metrics import ensure_series_axis
from glood.src.assessment.provider_utils import instantiate_providers

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_FONT_SIZE = 18
PLOT_TITLE_SIZE = 20
PLOT_LEGEND_SIZE = 16


FORCE_COMPONENTS = ("Fx_p", "Fy_p", "Fx_v_raw", "Fy_v_raw", "Fx_v", "Fy_v")
PLOT_COMPONENTS = ("Fx_p", "Fy_p", "Fx_v_raw", "Fy_v_raw")
TERCILE_LABELS = ("low", "mid", "high")


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Keep one compact table with mean/std for all relevant numeric outputs.
    cols = [
        col
        for col in df.columns
        if col.startswith("pred_")
        or col.startswith("true_")
        or col.startswith("err_")
        or col.startswith("rel_err_")
        or col.startswith("abs_rel_err_")
    ]
    # This table is intentionally simple: one row per scalar metric column.
    summary = df[cols].agg(["mean", "std"]).transpose().reset_index()
    summary.columns = ["metric", "mean", "std"]
    return summary


def _add_normalized_errors(df: pd.DataFrame, *, error_epsilon: float) -> pd.DataFrame:
    # Per-row normalized errors are the basis for robust cross-configuration comparisons.
    out = df.copy()
    for comp in FORCE_COMPONENTS:
        # Stabilized denominator avoids exploding ratios when |F_true| ~ 0.
        denom = np.abs(out[f"true_{comp}"].to_numpy(dtype=float)) + float(error_epsilon)
        err = out[f"err_{comp}"].to_numpy(dtype=float)
        rel = err / denom
        out[f"rel_err_{comp}"] = rel
        out[f"abs_rel_err_{comp}"] = np.abs(rel)
    return out


def _build_global(df: pd.DataFrame, *, viscous_prefactor: float, error_epsilon: float) -> pd.DataFrame:
    # Global table: component-wise aggregate scores over all assessed rows.
    rows = []
    for comp in FORCE_COMPONENTS:
        pred = df[f"pred_{comp}"].to_numpy(dtype=float)
        true = df[f"true_{comp}"].to_numpy(dtype=float)
        err = pred - true
        rel_err = df[f"rel_err_{comp}"].to_numpy(dtype=float)
        abs_rel_err = df[f"abs_rel_err_{comp}"].to_numpy(dtype=float)
        rows.append(
            {
                "component": comp,
                "viscous_prefactor": float(viscous_prefactor),
                "error_epsilon": float(error_epsilon),
                "n_rows": int(len(df)),
                "pred_sum": float(pred.sum()),
                "true_sum": float(true.sum()),
                "err_sum": float(err.sum()),
                "pred_mean": float(pred.mean()),
                "true_mean": float(true.mean()),
                "err_mean": float(err.mean()),
                "mae": float(abs(err).mean()),
                "rmse": float((err**2).mean() ** 0.5),
                "rel_err_mean": float(rel_err.mean()),
                "abs_rel_err_mean": float(abs_rel_err.mean()),
                "rel_rmse": float((rel_err**2).mean() ** 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values("component").reset_index(drop=True)


def _plot_example_series(df: pd.DataFrame, output_dir: Path, *, max_examples: int) -> None:
    # Plot a bounded number of examples to keep output size manageable.
    example_keys = (
        df[["example_id", "epoch"]]
        .drop_duplicates()
        .sort_values(["example_id", "epoch"])
        .head(max_examples)
        .itertuples(index=False, name=None)
    )
    for example_id, epoch in example_keys:
        sub = df[(df["example_id"] == example_id) & (df["epoch"] == epoch)].sort_values("timestep")
        if sub.empty:
            continue

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        for ax, comp in zip(axes.ravel(), PLOT_COMPONENTS):
            x = sub["timestep"].to_numpy()
            y_rel = sub[f"rel_err_{comp}"].to_numpy(dtype=float)
            ax.plot(x, y_rel, label="normalized error", linewidth=2)
            ax.axhline(0.0, color="k", linewidth=1, linestyle=":")
            ax.set_title(comp)
            ax.set_xlabel("timestep")
            ax.grid(True, alpha=0.3)

        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=1)
        fig.suptitle(f"Normalized Force Error vs Time | example={example_id}, epoch={epoch}")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = output_dir / f"example_{example_id:04d}_epoch_{epoch}_forces_normalized_error"
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _plot_error_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    # Signed normalized error distribution, pooled over examples and timesteps.
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df[f"rel_err_{comp}"].to_numpy(dtype=float) for comp in PLOT_COMPONENTS]
    ax.boxplot(data, tick_labels=list(PLOT_COMPONENTS), showfliers=False)
    ax.set_ylabel(r"relative error $(F_{pred}-F_{true})/(|F_{true}|+\epsilon)$", fontsize=PLOT_FONT_SIZE)
    ax.set_title("Normalized Force Error Distribution", fontsize=PLOT_TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "force_error_distribution_normalized"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_error_boxplot_exclusions(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    exclude_first_steps: tuple[int, ...],
) -> None:
    # Same distribution as above after dropping a prefix of early timesteps.
    for k in exclude_first_steps:
        if int(k) < 0:
            continue
        sub = df[df["timestep"] >= int(k)]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        data = [sub[f"rel_err_{comp}"].to_numpy(dtype=float) for comp in PLOT_COMPONENTS]
        ax.boxplot(data, tick_labels=list(PLOT_COMPONENTS), showfliers=False)
        ax.set_ylabel(r"relative error $(F_{pred}-F_{true})/(|F_{true}|+\epsilon)$", fontsize=PLOT_FONT_SIZE)
        ax.set_title(f"Normalized Force Error Distribution (exclude first {int(k)} timesteps)", fontsize=PLOT_TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = output_dir / f"force_error_distribution_normalized_excl_first_{int(k)}"
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _plot_mean_series(df: pd.DataFrame, output_dir: Path) -> None:
    # Timestep profile: mean +/- std of signed normalized error per component.
    grouped = df.groupby("timestep", dropna=False)
    mean_df = grouped.mean(numeric_only=True).reset_index()
    std_df = grouped.std(numeric_only=True).reset_index().fillna(0.0)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    for ax, comp in zip(axes.ravel(), PLOT_COMPONENTS):
        x = mean_df["timestep"].to_numpy(dtype=float)
        y = mean_df[f"rel_err_{comp}"].to_numpy(dtype=float)
        s = std_df[f"rel_err_{comp}"].to_numpy(dtype=float)

        ax.plot(x, y, label="mean normalized error", linewidth=2)
        ax.fill_between(x, y - s, y + s, alpha=0.2)
        ax.axhline(0.0, color="k", linewidth=1, linestyle=":")
        ax.set_title(comp)
        ax.set_xlabel("timestep")
        ax.grid(True, alpha=0.3)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=1)
    fig.suptitle("Mean +/- Std of Normalized Error vs Time")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = output_dir / "force_normalized_error_mean_series"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _compute_prefix_exclusion_stats(
    df: pd.DataFrame,
    *,
    components: tuple[str, ...],
) -> pd.DataFrame:
    # Quantify sensitivity to early-transient removal: keep rows with timestep >= k.
    # k=0 corresponds to using the full timeline.
    max_timestep = int(df["timestep"].max())
    rows: list[dict[str, float | int | str]] = []
    for comp in components:
        rel_col = f"rel_err_{comp}"
        abs_col = f"abs_rel_err_{comp}"
        for k in range(max_timestep + 1):
            sub = df[df["timestep"] >= k]
            if sub.empty:
                continue
            rel = sub[rel_col].to_numpy(dtype=float)
            abs_rel = sub[abs_col].to_numpy(dtype=float)
            rows.append(
                {
                    "component": comp,
                    "exclude_first_k": k,
                    "n_rows": int(len(sub)),
                    "rel_err_mean": float(rel.mean()),
                    "rel_err_std": float(rel.std(ddof=1)) if len(rel) > 1 else 0.0,
                    "abs_rel_err_mean": float(abs_rel.mean()),
                    "abs_rel_err_p05": float(np.quantile(abs_rel, 0.05)),
                    "abs_rel_err_p25": float(np.quantile(abs_rel, 0.25)),
                    "abs_rel_err_p50": float(np.quantile(abs_rel, 0.50)),
                    "abs_rel_err_median": float(np.median(abs_rel)),
                    "abs_rel_err_p75": float(np.quantile(abs_rel, 0.75)),
                    "abs_rel_err_p95": float(np.quantile(abs_rel, 0.95)),
                }
            )
    return pd.DataFrame(rows).sort_values(["component", "exclude_first_k"]).reset_index(drop=True)


def _plot_prefix_exclusion(stats_df: pd.DataFrame, output_dir: Path) -> None:
    # Visualize how robust percentile-based |rel err| statistics evolve as k increases.
    for comp in PLOT_COMPONENTS:
        sub = stats_df[stats_df["component"] == comp].sort_values("exclude_first_k")
        if sub.empty:
            continue
        x = sub["exclude_first_k"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, sub["abs_rel_err_p05"].to_numpy(dtype=float), label="p5 |rel err|", linewidth=2)
        ax.plot(x, sub["abs_rel_err_p25"].to_numpy(dtype=float), label="p25 |rel err|", linewidth=2)
        ax.plot(x, sub["abs_rel_err_p50"].to_numpy(dtype=float), label="p50 |rel err|", linewidth=2)
        ax.plot(x, sub["abs_rel_err_p75"].to_numpy(dtype=float), label="p75 |rel err|", linewidth=2)
        ax.plot(x, sub["abs_rel_err_p95"].to_numpy(dtype=float), label="p95 |rel err|", linewidth=2)
        ax.set_xlabel("excluded initial timesteps (k)", fontsize=PLOT_FONT_SIZE)
        ax.set_ylabel("|normalized error| statistic", fontsize=PLOT_FONT_SIZE)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=PLOT_LEGEND_SIZE)
        fig.tight_layout()
        out = output_dir / f"force_prefix_exclusion_{comp}"
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _compute_magnitude_gated_stats(
    df: pd.DataFrame,
    *,
    components: tuple[str, ...],
    true_force_percentiles: tuple[float, ...] | None,
    true_force_thresholds: tuple[float, ...] | None,
) -> pd.DataFrame:
    # Quantify denominator effects by gating on |F_true| > tau(p),
    # where tau(p) is the p-th percentile computed in the current reference set.
    rows: list[dict[str, float | int | str]] = []
    for comp in components:
        true_abs = np.abs(df[f"true_{comp}"].to_numpy(dtype=float))
        rel = df[f"rel_err_{comp}"].to_numpy(dtype=float)
        abs_rel = df[f"abs_rel_err_{comp}"].to_numpy(dtype=float)
        total_n = int(len(true_abs))
        if total_n == 0:
            continue
        if true_force_percentiles:
            percentile_thresholds: list[tuple[float, float]] = []
            for p in true_force_percentiles:
                p_clamped = float(np.clip(float(p), 0.0, 100.0))
                thr = float(np.quantile(true_abs, p_clamped / 100.0))
                percentile_thresholds.append((p_clamped, thr))
        elif true_force_thresholds:
            # Backward-compatible mode: infer percentile from user-supplied tau.
            percentile_thresholds = []
            for thr in true_force_thresholds:
                thr_val = float(thr)
                p_emp = float((true_abs <= thr_val).mean() * 100.0)
                percentile_thresholds.append((p_emp, thr_val))
        else:
            raise ValueError("Set analysis.true_force_percentiles (preferred) or analysis.true_force_thresholds.")

        for p_clamped, thr in percentile_thresholds:
            mask = true_abs > thr
            n = int(mask.sum())
            if n == 0:
                continue
            tau_percentile = float((true_abs <= thr).mean() * 100.0)
            rel_sub = rel[mask]
            abs_sub = abs_rel[mask]
            rows.append(
                {
                    "component": comp,
                    "target_percentile": p_clamped,
                    "true_force_threshold": float(thr),
                    "tau_percentile_empirical": tau_percentile,
                    "n_rows": n,
                    "retained_fraction": float(n / total_n) if total_n else 0.0,
                    "rel_err_mean": float(rel_sub.mean()),
                    "abs_rel_err_mean": float(abs_sub.mean()),
                    "abs_rel_err_median": float(np.median(abs_sub)),
                    "abs_rel_err_p75": float(np.quantile(abs_sub, 0.75)),
                    "abs_rel_err_p90": float(np.quantile(abs_sub, 0.90)),
                    "abs_rel_err_p95": float(np.quantile(abs_sub, 0.95)),
                }
            )
    return pd.DataFrame(rows).sort_values(["component", "target_percentile"]).reset_index(drop=True)


def _plot_magnitude_gated(stats_df: pd.DataFrame, output_dir: Path) -> None:
    # Use percentile as x-axis and annotate corresponding tau computed on current subset.
    for comp in PLOT_COMPONENTS:
        sub = stats_df[stats_df["component"] == comp].sort_values("target_percentile")
        if sub.empty:
            continue

        x = sub["target_percentile"].to_numpy(dtype=float)
        y_mean = sub["abs_rel_err_mean"].to_numpy(dtype=float)
        y_med = sub["abs_rel_err_median"].to_numpy(dtype=float)
        y_p75 = sub["abs_rel_err_p75"].to_numpy(dtype=float)
        y_p90 = sub["abs_rel_err_p90"].to_numpy(dtype=float)
        y_p95 = sub["abs_rel_err_p95"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y_mean, marker="o", linewidth=2, label="mean |rel err|")
        ax.plot(x, y_med, marker="o", linewidth=2, label="median |rel err|")
        ax.plot(x, y_p75, marker="o", linewidth=2, label="p75 |rel err|")
        ax.plot(x, y_p90, marker="o", linewidth=2, label="p90 |rel err|")
        ax.plot(x, y_p95, marker="o", linewidth=2, label="p95 |rel err|")
        ax.set_title(f"Magnitude-Gated Sensitivity ({comp})")
        ax.set_xlabel("gating percentile p")
        ax.set_ylabel("|normalized error| statistic")
        tau = sub["true_force_threshold"].to_numpy(dtype=float)
        ax.set_xticks(x)
        # Tick label reports percentile first, then tau value computed in this subset.
        ax.set_xticklabels([f"{p:.0f}%\n($\\tau$={t:.0e})" for p, t in zip(x, tau)])
        ax.set_xlabel("gating percentile p\n(corresponding tau computed in current reference set)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = output_dir / f"force_magnitude_gated_{comp}"
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _compute_example_last_timestep_scores(df: pd.DataFrame) -> pd.DataFrame:
    # One score row per (example_id, epoch), based on true forces at final timestep.
    if df.empty:
        return pd.DataFrame(
            columns=[
                "example_id",
                "epoch",
                "last_timestep",
                "score_pressure",
                "score_viscous_raw",
            ]
        )
    max_timestep = df.groupby(["example_id", "epoch"], as_index=False)["timestep"].max()
    max_timestep = max_timestep.rename(columns={"timestep": "last_timestep"})
    merged = df.merge(
        max_timestep,
        left_on=["example_id", "epoch", "timestep"],
        right_on=["example_id", "epoch", "last_timestep"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Failed to recover last timestep rows for stratification.")

    out = merged[["example_id", "epoch", "last_timestep"]].copy()
    out["score_pressure"] = (
        merged["true_Fx_p"].to_numpy(dtype=float) ** 2 + merged["true_Fy_p"].to_numpy(dtype=float) ** 2
    )
    out["score_viscous_raw"] = (
        merged["true_Fx_v_raw"].to_numpy(dtype=float) ** 2 + merged["true_Fy_v_raw"].to_numpy(dtype=float) ** 2
    )
    return out


def _assign_terciles(scores: pd.Series) -> pd.Series:
    # Rank-based qcut avoids issues when many values are tied.
    n = int(scores.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 examples to build terciles.")
    ranked = scores.rank(method="first")
    bins = pd.qcut(ranked, q=3, labels=list(TERCILE_LABELS))
    return bins.astype(str)


def _run_analysis_bundle(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    output_stem_prefix: str,
    plotting_enabled: bool,
    plotting_subdir: str,
    max_examples: int,
    error_boxplot_exclusions: tuple[int, ...],
    true_force_percentiles: tuple[float, ...] | None,
    true_force_thresholds: tuple[float, ...] | None,
    viscous_prefactor: float,
    error_epsilon: float,
) -> None:
    summary_df = _build_summary(df)
    global_df = _build_global(
        df,
        viscous_prefactor=viscous_prefactor,
        error_epsilon=error_epsilon,
    )
    prefix_df = _compute_prefix_exclusion_stats(df, components=PLOT_COMPONENTS)
    magnitude_df = _compute_magnitude_gated_stats(
        df,
        components=PLOT_COMPONENTS,
        true_force_percentiles=true_force_percentiles,
        true_force_thresholds=true_force_thresholds,
    )

    summary_df.to_csv(output_dir / f"{output_stem_prefix}summary.csv", index=False)
    global_df.to_csv(output_dir / f"{output_stem_prefix}global.csv", index=False)
    prefix_df.to_csv(output_dir / f"{output_stem_prefix}prefix_exclusion.csv", index=False)
    magnitude_df.to_csv(output_dir / f"{output_stem_prefix}magnitude_gated.csv", index=False)

    if plotting_enabled:
        plot_dir = output_dir / plotting_subdir
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_mean_series(df, plot_dir)
        _plot_error_boxplot(df, plot_dir)
        _plot_error_boxplot_exclusions(
            df,
            plot_dir,
            exclude_first_steps=error_boxplot_exclusions,
        )
        _plot_example_series(
            df,
            plot_dir,
            max_examples=max_examples,
        )
        _plot_prefix_exclusion(prefix_df, plot_dir)
        _plot_magnitude_gated(magnitude_df, plot_dir)


@hydra.main(version_base=None, config_path="conf", config_name="assess_drag_force_generic")
def main(cfg: DictConfig) -> None:
    with (env.out() / "conf_assess_drag_force.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Reuse plotting providers so assessment runs on the same data-selection
    # logic used by existing post-processing entry points.
    providers = instantiate_providers(cfg.plotting_providers)

    rows: list[dict[str, float | int]] = []
    assessed_examples = 0

    viscous_prefactor = float(cfg.assessment.viscous_prefactor)
    p_channel = int(cfg.assessment.p_channel)
    ux_channel = int(cfg.assessment.ux_channel)
    uy_channel = int(cfg.assessment.uy_channel)
    fluid_value = int(cfg.assessment.fluid_value)
    wall_value = int(cfg.assessment.wall_value)
    crop_margin = int(cfg.assessment.crop_margin)
    dx = float(cfg.assessment.dx)
    dy = float(cfg.assessment.dy)
    error_epsilon = float(cfg.assessment.error_epsilon)
    require_matnum = bool(cfg.assessment.require_matnum)

    # Providers yield per-example tensors and optional matnum maps from chosen input source.
    for provider in providers:
        for plottable_input in provider:
            if not hasattr(plottable_input, "pred") or not hasattr(plottable_input, "true"):
                raise TypeError("Plotting provider must yield objects with 'pred' and 'true' attributes.")

            if plottable_input.matnum is None:
                if require_matnum:
                    raise ValueError(
                        f"matnum is required but missing for example {plottable_input.id}, epoch {plottable_input.epoch}."
                    )
                continue

            pred = ensure_series_axis(plottable_input.pred, name="pred")
            true = ensure_series_axis(plottable_input.true, name="true")
            if pred.shape != true.shape:
                raise ValueError(
                    f"Pred/true shape mismatch for example {plottable_input.id}: {pred.shape} vs {true.shape}."
                )

            # Compute pred/true force components using identical geometry/material masks.
            pred_series = compute_series_force_components(
                pred,
                plottable_input.matnum,
                p_channel=p_channel,
                ux_channel=ux_channel,
                uy_channel=uy_channel,
                fluid_value=fluid_value,
                wall_value=wall_value,
                crop_margin=crop_margin,
                dx=dx,
                dy=dy,
                viscous_prefactor=viscous_prefactor,
            )
            true_series = compute_series_force_components(
                true,
                plottable_input.matnum,
                p_channel=p_channel,
                ux_channel=ux_channel,
                uy_channel=uy_channel,
                fluid_value=fluid_value,
                wall_value=wall_value,
                crop_margin=crop_margin,
                dx=dx,
                dy=dy,
                viscous_prefactor=viscous_prefactor,
            )

            # Persist raw forces per timestep; normalized errors are derived later.
            for timestep, (pred_comp, true_comp) in enumerate(zip(pred_series, true_series)):
                row: dict[str, float | int] = {
                    "example_id": int(plottable_input.id),
                    "epoch": int(plottable_input.epoch),
                    "timestep": int(timestep),
                    "viscous_prefactor": viscous_prefactor,
                    "pred_interface_faces": int(pred_comp["interface_faces"]),
                    "true_interface_faces": int(true_comp["interface_faces"]),
                }
                for comp in FORCE_COMPONENTS:
                    p_val = float(pred_comp[comp])
                    t_val = float(true_comp[comp])
                    row[f"pred_{comp}"] = p_val
                    row[f"true_{comp}"] = t_val
                    row[f"err_{comp}"] = p_val - t_val
                rows.append(row)

            assessed_examples += 1

    if not rows:
        raise ValueError("No examples were assessed. Check provider/input configuration.")

    output_dir = Path(env.out())
    output_dir.mkdir(parents=True, exist_ok=True)

    per_timestep_df = pd.DataFrame(rows)
    # Derived dataframe used only for aggregate statistics and diagnostics.
    # Raw per-timestep values remain in `per_timestep_df` for downstream reuse.
    analysis_df = _add_normalized_errors(per_timestep_df, error_epsilon=error_epsilon)
    per_timestep_path = output_dir / str(cfg.outputs.per_timestep_csv)
    per_timestep_df.to_csv(per_timestep_path, index=False)
    output_stem_prefix = "drag_force_"
    true_force_percentiles = (
        tuple(float(x) for x in cfg.analysis.true_force_percentiles)
        if "true_force_percentiles" in cfg.analysis
        else None
    )
    true_force_thresholds = (
        tuple(float(x) for x in cfg.analysis.true_force_thresholds)
        if "true_force_thresholds" in cfg.analysis
        else None
    )
    _run_analysis_bundle(
        analysis_df,
        output_dir=output_dir,
        output_stem_prefix=output_stem_prefix,
        plotting_enabled=bool(cfg.plotting.enabled),
        plotting_subdir=str(cfg.plotting.subdir),
        max_examples=int(cfg.plotting.max_examples),
        error_boxplot_exclusions=tuple(int(x) for x in cfg.plotting.error_boxplot_exclusions),
        true_force_percentiles=true_force_percentiles,
        true_force_thresholds=true_force_thresholds,
        viscous_prefactor=viscous_prefactor,
        error_epsilon=error_epsilon,
    )

    # Stratified analysis by per-example last-timestep true-force magnitudes.
    score_df = _compute_example_last_timestep_scores(per_timestep_df)
    score_df["tercile_pressure"] = _assign_terciles(score_df["score_pressure"])
    score_df["tercile_viscous_raw"] = _assign_terciles(score_df["score_viscous_raw"])
    score_df.to_csv(output_dir / "drag_force_example_terciles.csv", index=False)

    for strat_col, strat_name in (
        ("tercile_pressure", "pressure"),
        ("tercile_viscous_raw", "viscous_raw"),
    ):
        for tercile in TERCILE_LABELS:
            sub_keys = score_df[score_df[strat_col] == tercile][["example_id", "epoch"]].drop_duplicates()
            if sub_keys.empty:
                continue
            sub_raw_df = per_timestep_df.merge(sub_keys, on=["example_id", "epoch"], how="inner")
            if sub_raw_df.empty:
                continue
            sub_df = analysis_df.merge(sub_keys, on=["example_id", "epoch"], how="inner")
            sub_dir = output_dir / "stratified" / strat_name / tercile
            sub_dir.mkdir(parents=True, exist_ok=True)
            sub_raw_df.to_csv(sub_dir / "drag_force_per_timestep.csv", index=False)
            _run_analysis_bundle(
                sub_df,
                output_dir=sub_dir,
                output_stem_prefix="drag_force_",
                plotting_enabled=bool(cfg.plotting.enabled),
                plotting_subdir=str(cfg.plotting.subdir),
                max_examples=int(cfg.plotting.max_examples),
                error_boxplot_exclusions=tuple(int(x) for x in cfg.plotting.error_boxplot_exclusions),
                true_force_percentiles=true_force_percentiles,
                true_force_thresholds=true_force_thresholds,
                viscous_prefactor=viscous_prefactor,
                error_epsilon=error_epsilon,
            )

    print(f"Assessed examples: {assessed_examples}")
    print(f"Per-timestep forces -> {per_timestep_path.name}")
    print("Summary stats       -> drag_force_summary.csv")
    print("Global stats        -> drag_force_global.csv")
    print("Prefix exclusion    -> drag_force_prefix_exclusion.csv")
    print("Magnitude-gated     -> drag_force_magnitude_gated.csv")
    print("Example terciles    -> drag_force_example_terciles.csv")
    print("Stratified outputs  -> stratified/pressure/{low,mid,high} and stratified/viscous_raw/{low,mid,high}")
    if bool(cfg.plotting.enabled):
        print(f"Plots               -> {output_dir / str(cfg.plotting.subdir)}")


if __name__ == "__main__":
    main()
