from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env
from glood.src.assessment.input_materialization import materialized_inference_input

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _iter_example_dirs(input_dir: Path):
    data_dirs: list[tuple[int, int, Path]] = []
    for data_dir in input_dir.glob("example_*/epoch_*/data"):
        example_dir = data_dir.parent.parent
        epoch_dir = data_dir.parent
        example_id = int(example_dir.name.split("_", 1)[1])
        epoch = int(epoch_dir.name.split("_", 1)[1])
        data_dirs.append((example_id, epoch, data_dir))

    for example_id, epoch, data_dir in sorted(data_dirs):
        yield example_id, epoch, data_dir


def _ensure_series_axis(array: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim < 3:
        raise ValueError(f"{name} must have at least 3 dimensions, got {array.shape}.")
    if array.ndim == 3:
        return np.expand_dims(array, axis=0)
    return array


def _normalize_matnum_series(matnum: np.ndarray, *, series_len: int) -> np.ndarray:
    matnum = np.asarray(matnum)
    if matnum.ndim == 4 and matnum.shape[-1] == 1:
        matnum = matnum[..., 0]
    elif matnum.ndim == 2:
        matnum = np.expand_dims(matnum, axis=0)
    elif matnum.ndim not in (3, 4):
        raise ValueError(f"Unsupported matnum shape: {matnum.shape}")

    if matnum.ndim != 3:
        raise ValueError(f"matnum must resolve to (T,H,W), got {matnum.shape}")
    if matnum.shape[0] == 1 and series_len > 1:
        matnum = np.repeat(matnum, series_len, axis=0)
    if matnum.shape[0] != series_len:
        raise ValueError(f"matnum series length mismatch: expected {series_len}, got {matnum.shape[0]}")
    return matnum


def _resolve_section_indices(width: int, fractions: tuple[float, ...]) -> list[tuple[float, int]]:
    indices: list[tuple[float, int]] = []
    seen: set[int] = set()
    for fraction in fractions:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Section fraction must be in [0,1], got {value}.")
        x_index = int(round(value * (width - 1)))
        x_index = int(np.clip(x_index, 0, width - 1))
        if x_index in seen:
            continue
        indices.append((value, x_index))
        seen.add(x_index)
    if not indices:
        raise ValueError("No section indices resolved from x_fractions.")
    return indices


def _relative_l2_error(pred_profile: np.ndarray, true_profile: np.ndarray, *, eps: float) -> float:
    pred_profile = np.asarray(pred_profile, dtype=float)
    true_profile = np.asarray(true_profile, dtype=float)
    numerator = np.linalg.norm(pred_profile - true_profile)
    denominator = np.linalg.norm(true_profile)
    return float(numerator / (denominator + float(eps)))


def _build_summary(per_profile_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = (
        per_profile_df.groupby(["model", "geometry", "dataset_label", "x_fraction", "x_index"], dropna=False)
        .agg(
            n_profiles=("profile_error", "size"),
            n_examples=("example_id", "nunique"),
            n_epochs=("epoch", "nunique"),
            median_profile_error=("profile_error", "median"),
            q25_profile_error=("profile_error", lambda s: float(np.quantile(s, 0.25))),
            q75_profile_error=("profile_error", lambda s: float(np.quantile(s, 0.75))),
            median_valid_points=("n_valid_points", "median"),
            q25_valid_points=("n_valid_points", lambda s: float(np.quantile(s, 0.25))),
            q75_valid_points=("n_valid_points", lambda s: float(np.quantile(s, 0.75))),
            min_valid_points=("n_valid_points", "min"),
        )
        .reset_index()
        .sort_values(["model", "geometry", "x_fraction"])
        .reset_index(drop=True)
    )
    summary_df["iqr_profile_error"] = summary_df["q75_profile_error"] - summary_df["q25_profile_error"]

    overall_df = (
        per_profile_df.groupby(["model", "geometry", "dataset_label"], dropna=False)
        .agg(
            n_profiles=("profile_error", "size"),
            n_examples=("example_id", "nunique"),
            median_profile_error=("profile_error", "median"),
            q25_profile_error=("profile_error", lambda s: float(np.quantile(s, 0.25))),
            q75_profile_error=("profile_error", lambda s: float(np.quantile(s, 0.75))),
        )
        .reset_index()
        .sort_values(["model", "geometry"])
        .reset_index(drop=True)
    )
    overall_df["iqr_profile_error"] = overall_df["q75_profile_error"] - overall_df["q25_profile_error"]
    return summary_df, overall_df


def _compute_prefix_exclusion_stats(per_profile_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (model, geometry, dataset_label), sub_df in per_profile_df.groupby(
        ["model", "geometry", "dataset_label"], dropna=False
    ):
        max_timestep = int(sub_df["timestep"].max())
        for k in range(max_timestep + 1):
            trimmed = sub_df[sub_df["timestep"] >= k]
            if trimmed.empty:
                continue
            errors = trimmed["profile_error"].to_numpy(dtype=float)
            true_norms = trimmed["true_profile_l2"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": str(model),
                    "geometry": str(geometry),
                    "dataset_label": str(dataset_label),
                    "exclude_first_k": int(k),
                    "n_profiles": int(len(trimmed)),
                    "n_examples": int(trimmed["example_id"].nunique()),
                    "median_profile_error": float(np.median(errors)),
                    "q25_profile_error": float(np.quantile(errors, 0.25)),
                    "q75_profile_error": float(np.quantile(errors, 0.75)),
                    "iqr_profile_error": float(np.quantile(errors, 0.75) - np.quantile(errors, 0.25)),
                    "median_true_profile_l2": float(np.median(true_norms)),
                    "frac_true_profile_l2_zero": float((true_norms <= 1.0e-12).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "geometry", "exclude_first_k"]).reset_index(drop=True)


def _compute_true_profile_l2_gated_stats(
    per_profile_df: pd.DataFrame,
    *,
    true_profile_l2_percentiles: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (model, geometry, dataset_label), sub_df in per_profile_df.groupby(
        ["model", "geometry", "dataset_label"], dropna=False
    ):
        true_norms = sub_df["true_profile_l2"].to_numpy(dtype=float)
        errors = sub_df["profile_error"].to_numpy(dtype=float)
        total_n = int(len(sub_df))
        for percentile in true_profile_l2_percentiles:
            p_clamped = float(np.clip(float(percentile), 0.0, 100.0))
            threshold = float(np.quantile(true_norms, p_clamped / 100.0))
            mask = true_norms > threshold
            retained = sub_df[mask]
            if retained.empty:
                continue
            retained_errors = retained["profile_error"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": str(model),
                    "geometry": str(geometry),
                    "dataset_label": str(dataset_label),
                    "target_percentile": p_clamped,
                    "true_profile_l2_threshold": threshold,
                    "n_profiles": int(len(retained)),
                    "retained_fraction": float(len(retained) / total_n) if total_n else 0.0,
                    "median_profile_error": float(np.median(retained_errors)),
                    "q25_profile_error": float(np.quantile(retained_errors, 0.25)),
                    "q75_profile_error": float(np.quantile(retained_errors, 0.75)),
                    "iqr_profile_error": float(np.quantile(retained_errors, 0.75) - np.quantile(retained_errors, 0.25)),
                    "median_true_profile_l2": float(np.median(retained["true_profile_l2"].to_numpy(dtype=float))),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "geometry", "target_percentile"]).reset_index(drop=True)


def _compute_true_profile_l2_gated_by_timestep_stats(
    per_profile_df: pd.DataFrame,
    *,
    true_profile_l2_percentiles: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (model, geometry, dataset_label, timestep), sub_df in per_profile_df.groupby(
        ["model", "geometry", "dataset_label", "timestep"], dropna=False
    ):
        true_norms = sub_df["true_profile_l2"].to_numpy(dtype=float)
        total_n = int(len(sub_df))
        for percentile in true_profile_l2_percentiles:
            p_clamped = float(np.clip(float(percentile), 0.0, 100.0))
            threshold = float(np.quantile(true_norms, p_clamped / 100.0))
            retained = sub_df[sub_df["true_profile_l2"] > threshold]
            if retained.empty:
                continue
            retained_errors = retained["profile_error"].to_numpy(dtype=float)
            retained_true_norms = retained["true_profile_l2"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": str(model),
                    "geometry": str(geometry),
                    "dataset_label": str(dataset_label),
                    "timestep": int(timestep),
                    "target_percentile": p_clamped,
                    "true_profile_l2_threshold": threshold,
                    "n_profiles": int(len(retained)),
                    "retained_fraction": float(len(retained) / total_n) if total_n else 0.0,
                    "median_profile_error": float(np.median(retained_errors)),
                    "q25_profile_error": float(np.quantile(retained_errors, 0.25)),
                    "q75_profile_error": float(np.quantile(retained_errors, 0.75)),
                    "iqr_profile_error": float(np.quantile(retained_errors, 0.75) - np.quantile(retained_errors, 0.25)),
                    "median_true_profile_l2": float(np.median(retained_true_norms)),
                    "frac_true_profile_l2_zero": float((retained_true_norms <= 1.0e-12).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "geometry", "timestep", "target_percentile"]).reset_index(drop=True)


def _build_reference_timesteps(series_len: int) -> tuple[int, ...]:
    raw = (0, series_len // 2, series_len - 1)
    unique: list[int] = []
    seen: set[int] = set()
    for timestep in raw:
        if timestep not in seen:
            unique.append(int(timestep))
            seen.add(int(timestep))
    return tuple(unique)


def _pick_representative_example_ids(example_ids: tuple[int, ...], *, count: int) -> tuple[int, ...]:
    if count <= 0 or not example_ids:
        return ()
    if len(example_ids) <= count:
        return example_ids

    positions = np.linspace(0, len(example_ids) - 1, num=count)
    selected: list[int] = []
    seen: set[int] = set()
    for position in positions:
        index = int(round(float(position)))
        example_id = int(example_ids[index])
        if example_id in seen:
            continue
        selected.append(example_id)
        seen.add(example_id)
    return tuple(selected)


def _resolve_input_dir(input_path: Path, *, label: str) -> Path:
    return input_path / "infer"


def _format_section_label(x_fraction: float) -> str:
    return f"x/L={float(x_fraction):.2f}"


def _boxplot_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="C0", linewidth=2, label="median"),
        Line2D([0], [0], color="C0", linewidth=8, alpha=0.25, label="IQR box"),
        Line2D([0], [0], color="C0", linewidth=1, linestyle="--", label="whiskers (1.5 IQR)"),
    ]


def _compute_shared_boxplot_ylim(summary_df: pd.DataFrame) -> tuple[float, float]:
    whisker_top = summary_df["q75_profile_error"] + 1.5 * summary_df["iqr_profile_error"]
    y_max = float(whisker_top.max()) if not whisker_top.empty else 1.0
    y_max = max(y_max, 1.0e-6)
    return 0.0, y_max * 1.05


def _plot_boxplot_panel(
    *,
    plot_df: pd.DataFrame,
    ordered_pairs: list[tuple[str, str]],
    title: str,
    output_stem: Path,
    y_limits: tuple[float, float],
    log_scale: bool,
) -> None:
    labels = [f"{dataset_label}\n{section}" for dataset_label, section in ordered_pairs]
    data = [
        plot_df[(plot_df["dataset_label"] == dataset_label) & (plot_df["section_label"] == section)]["profile_error"]
        .to_numpy(dtype=float)
        for dataset_label, section in ordered_pairs
    ]

    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(labels)), 6))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel(r"relative $L^2$ profile error")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(handles=_boxplot_legend_handles(), loc="upper right", frameon=False)

    if log_scale:
        positive_values = np.concatenate([values[values > 0.0] for values in data if len(values)] or [np.array([1.0e-8])])
        y_min = max(float(np.min(positive_values)), 1.0e-8)
        y_max = max(float(y_limits[1]), y_min * 10.0)
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(*y_limits)

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_boxplot(summary_df: pd.DataFrame, per_profile_df: pd.DataFrame, output_dir: Path, *, stem: str) -> None:
    plot_df = per_profile_df.copy()
    plot_df["section_label"] = plot_df["x_fraction"].map(_format_section_label)
    summary_with_labels = summary_df.assign(section_label=lambda df: df["x_fraction"].map(_format_section_label))
    shared_y_limits = _compute_shared_boxplot_ylim(summary_df)

    ordered_pairs_all = [
        (row.dataset_label, row.section_label)
        for row in summary_with_labels[["dataset_label", "section_label"]].drop_duplicates().itertuples(index=False)
    ]
    _plot_boxplot_panel(
        plot_df=plot_df,
        ordered_pairs=ordered_pairs_all,
        title=r"$u_x(y)$ profile error on fixed sections",
        output_stem=output_dir / stem,
        y_limits=shared_y_limits,
        log_scale=False,
    )
    _plot_boxplot_panel(
        plot_df=plot_df,
        ordered_pairs=ordered_pairs_all,
        title=r"$u_x(y)$ profile error on fixed sections (log y)",
        output_stem=output_dir / f"{stem}_log",
        y_limits=shared_y_limits,
        log_scale=True,
    )

    for geometry, geometry_df in summary_with_labels.groupby("geometry", sort=True):
        ordered_pairs = [
            (row.dataset_label, row.section_label)
            for row in geometry_df[["dataset_label", "section_label"]].drop_duplicates().itertuples(index=False)
        ]
        geometry_plot_df = plot_df[plot_df["geometry"] == geometry]
        geometry_title = rf"$u_x(y)$ profile error on fixed sections | {geometry}"
        _plot_boxplot_panel(
            plot_df=geometry_plot_df,
            ordered_pairs=ordered_pairs,
            title=geometry_title,
            output_stem=output_dir / f"{stem}_{geometry}",
            y_limits=shared_y_limits,
            log_scale=False,
        )
        _plot_boxplot_panel(
            plot_df=geometry_plot_df,
            ordered_pairs=ordered_pairs,
            title=f"{geometry_title} (log y)",
            output_stem=output_dir / f"{stem}_{geometry}_log",
            y_limits=shared_y_limits,
            log_scale=True,
        )


def _collect_selected_profile_rows(
    *,
    dataset_label: str,
    example_id: int,
    epoch: int,
    timestep: int,
    x_fraction: float,
    x_index: int,
    y_indices: np.ndarray,
    true_line: np.ndarray,
    pred_line: np.ndarray,
    channels: dict[str, int],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    variable_map = {
        "p": (channels["p"], true_line[:, channels["p"]], pred_line[:, channels["p"]]),
        "ux": (channels["ux"], true_line[:, channels["ux"]], pred_line[:, channels["ux"]]),
        "uy": (channels["uy"], true_line[:, channels["uy"]], pred_line[:, channels["uy"]]),
    }
    for variable_name, (channel_index, true_values, pred_values) in variable_map.items():
        for y_index, true_value, pred_value in zip(y_indices, true_values, pred_values):
            rows.append(
                {
                    "dataset": dataset_label,
                    "example_id": int(example_id),
                    "epoch": int(epoch),
                    "timestep": int(timestep),
                    "x_fraction": float(x_fraction),
                    "x_index": int(x_index),
                    "y_index": int(y_index),
                    "variable": variable_name,
                    "channel_index": int(channel_index),
                    "true_value": float(true_value),
                    "pred_value": float(pred_value),
                }
            )
    return rows


def _plot_selected_example_profiles(
    selected_profiles_df: pd.DataFrame,
    output_dir: Path,
    *,
    subdir: str,
    include_column_index: bool,
) -> None:
    selected_dir = output_dir / subdir
    selected_dir.mkdir(parents=True, exist_ok=True)
    variable_order = ("p", "ux", "uy")

    for (dataset_label, example_id, timestep), sub_df in selected_profiles_df.groupby(
        ["dataset", "example_id", "timestep"], dropna=False
    ):
        sections = list(
            sub_df[["x_fraction", "x_index"]]
            .drop_duplicates()
            .sort_values("x_fraction")
            .itertuples(index=False, name=None)
        )
        fig, axes = plt.subplots(nrows=len(variable_order), ncols=len(sections), figsize=(12, 8), sharex=False)

        for row_idx, variable_name in enumerate(variable_order):
            for col_idx, (x_fraction, x_index) in enumerate(sections):
                ax = axes[row_idx, col_idx]
                panel_df = sub_df[(sub_df["variable"] == variable_name) & (sub_df["x_index"] == x_index)].sort_values(
                    "y_index"
                )
                ax.plot(panel_df["y_index"], panel_df["true_value"], label="true", linewidth=2)
                ax.plot(panel_df["y_index"], panel_df["pred_value"], label="pred", linewidth=2, linestyle="--")
                if row_idx == 0:
                    title = f"x/L={float(x_fraction):.2f}"
                    if include_column_index:
                        title = f"{title} (col {int(x_index)})"
                    ax.set_title(title)
                if col_idx == 0:
                    ax.set_ylabel(variable_name)
                if row_idx == len(variable_order) - 1:
                    ax.set_xlabel("y index")
                ax.grid(True, alpha=0.3)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.975), frameon=False)
        fig.suptitle(f"{dataset_label} | example {int(example_id)} | timestep {int(timestep)}", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        stem = selected_dir / f"{dataset_label}_example_{int(example_id):03d}_timestep_{int(timestep):02d}_line_profiles"
        fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="assess_profiles_generic")
def main(cfg: DictConfig) -> None:
    out_dir = env.out()
    with (out_dir / "conf_assess_profiles.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    x_fractions = tuple(float(x) for x in cfg.analysis.x_fractions)
    selected_examples = {
        str(geometry): tuple(int(x) for x in ids)
        for geometry, ids in OmegaConf.to_container(cfg.analysis.selected_examples, resolve=True).items()
    }
    selected_examples_count = int(getattr(cfg.analysis, "selected_examples_count", 3))
    channels = {
        "p": int(cfg.analysis.p_channel),
        "ux": int(cfg.analysis.ux_channel),
        "uy": int(cfg.analysis.uy_channel),
    }
    fluid_value = int(cfg.analysis.fluid_value)
    epsilon = float(cfg.analysis.error_epsilon)

    manifest_rows: list[dict[str, int | str]] = []
    rows: list[dict[str, float | int | str]] = []
    selected_profile_rows: list[dict[str, float | int | str]] = []

    for dataset_cfg in cfg.datasets:
        model_name = str(dataset_cfg.model)
        geometry = str(dataset_cfg.geometry)
        dataset_label = str(dataset_cfg.label)
        input_path = Path(str(dataset_cfg.input_dir))
        selected_ids = set(int(x) for x in selected_examples.get(geometry, ()))

        example_count = 0
        with materialized_inference_input(input_path, label=dataset_label, cfg=cfg) as input_dir:
            example_batches = list(_iter_example_dirs(input_dir))
            if not selected_ids:
                available_example_ids = tuple(sorted({int(example_id) for example_id, _, _ in example_batches}))
                selected_ids = set(_pick_representative_example_ids(available_example_ids, count=selected_examples_count))
            for example_id in sorted(selected_ids):
                manifest_rows.append({"geometry": geometry, "dataset_label": dataset_label, "example_id": example_id})

            for example_id, epoch, data_dir in example_batches:
                example_count += 1
                pred = _ensure_series_axis(
                    np.load(data_dir / "pred.npy", allow_pickle=False, mmap_mode="r"),
                    name="pred",
                )
                true = _ensure_series_axis(
                    np.load(data_dir / "true.npy", allow_pickle=False, mmap_mode="r"),
                    name="true",
                )
                if pred.shape != true.shape:
                    raise ValueError(
                        f"Pred/true shape mismatch for dataset={dataset_label}, example={example_id}: {pred.shape} vs {true.shape}"
                    )
                if pred.ndim != 4:
                    raise ValueError(f"Expected (T,H,W,C) arrays, got {pred.shape}")

                matnum_path = data_dir / "matnum.npy"
                if not matnum_path.is_file():
                    raise ValueError(f"Missing matnum.npy for dataset={dataset_label}, example={example_id}")
                matnum = _normalize_matnum_series(
                    np.load(matnum_path, allow_pickle=False, mmap_mode="r"),
                    series_len=pred.shape[0],
                )
                fluid_mask = matnum == fluid_value
                sections = _resolve_section_indices(pred.shape[2], x_fractions)
                selected_timesteps = _build_reference_timesteps(pred.shape[0]) if example_id in selected_ids else ()

                for timestep in range(pred.shape[0]):
                    pred_ux = pred[timestep, :, :, channels["ux"]]
                    true_ux = true[timestep, :, :, channels["ux"]]
                    timestep_mask = fluid_mask[timestep]

                    for x_fraction, x_index in sections:
                        valid_mask = timestep_mask[:, x_index]
                        n_valid_points = int(valid_mask.sum())
                        if n_valid_points == 0:
                            continue

                        pred_profile = pred_ux[:, x_index][valid_mask]
                        true_profile = true_ux[:, x_index][valid_mask]
                        rows.append(
                            {
                                "model": model_name,
                                "geometry": geometry,
                                "dataset_label": dataset_label,
                                "example_id": int(example_id),
                                "epoch": int(epoch),
                                "timestep": int(timestep),
                                "x_fraction": float(x_fraction),
                                "x_index": int(x_index),
                                "n_valid_points": n_valid_points,
                                "true_profile_l2": float(np.linalg.norm(true_profile)),
                                "pred_profile_l2": float(np.linalg.norm(pred_profile)),
                                "profile_error": _relative_l2_error(pred_profile, true_profile, eps=epsilon),
                            }
                        )

                        if timestep in selected_timesteps:
                            y_indices = np.flatnonzero(valid_mask)
                            pred_line = pred[timestep, y_indices, x_index, :]
                            true_line = true[timestep, y_indices, x_index, :]
                            selected_profile_rows.extend(
                                _collect_selected_profile_rows(
                                    dataset_label=dataset_label,
                                    example_id=example_id,
                                    epoch=epoch,
                                    timestep=timestep,
                                    x_fraction=x_fraction,
                                    x_index=x_index,
                                    y_indices=y_indices,
                                    true_line=true_line,
                                    pred_line=pred_line,
                                    channels=channels,
                                )
                            )

        print(f"{dataset_label}: processed {example_count} examples from {input_path}")

    if not rows:
        raise ValueError("No profile rows were computed.")

    per_profile_df = pd.DataFrame(rows).sort_values(
        ["dataset_label", "example_id", "epoch", "timestep", "x_fraction"]
    ).reset_index(drop=True)
    summary_df, overall_df = _build_summary(per_profile_df)
    selected_profiles_columns = [
        "dataset",
        "example_id",
        "epoch",
        "timestep",
        "x_fraction",
        "x_index",
        "y_index",
        "variable",
        "channel_index",
        "true_value",
        "pred_value",
    ]
    selected_profiles_df = pd.DataFrame(selected_profile_rows, columns=selected_profiles_columns)
    if not selected_profiles_df.empty:
        selected_profiles_df = selected_profiles_df.sort_values(
            ["dataset", "example_id", "timestep", "x_fraction", "variable", "y_index"]
        ).reset_index(drop=True)
    manifest_df = pd.DataFrame(manifest_rows).drop_duplicates().sort_values(
        ["geometry", "dataset_label", "example_id"]
    ).reset_index(drop=True)
    prefix_df = _compute_prefix_exclusion_stats(per_profile_df)
    gated_df = _compute_true_profile_l2_gated_stats(
        per_profile_df,
        true_profile_l2_percentiles=tuple(float(x) for x in cfg.analysis.true_profile_l2_percentiles),
    )
    gated_by_timestep_df = _compute_true_profile_l2_gated_by_timestep_stats(
        per_profile_df,
        true_profile_l2_percentiles=tuple(float(x) for x in cfg.analysis.true_profile_l2_percentiles),
    )

    manifest_df.to_csv(out_dir / cfg.outputs.selected_examples_manifest_csv, index=False)
    per_profile_df.to_csv(out_dir / cfg.outputs.per_profile_csv, index=False)
    summary_df.to_csv(out_dir / cfg.outputs.summary_csv, index=False)
    overall_df.to_csv(out_dir / cfg.outputs.overall_csv, index=False)
    selected_profiles_df.to_csv(out_dir / cfg.outputs.selected_profiles_csv, index=False)
    prefix_df.to_csv(out_dir / cfg.outputs.prefix_exclusion_csv, index=False)
    gated_df.to_csv(out_dir / cfg.outputs.true_profile_l2_gated_csv, index=False)
    gated_by_timestep_df.to_csv(out_dir / cfg.outputs.true_profile_l2_gated_by_timestep_csv, index=False)

    if bool(cfg.plotting.enabled):
        _plot_boxplot(summary_df, per_profile_df, out_dir, stem=str(cfg.plotting.boxplot_stem))
        _plot_selected_example_profiles(
            selected_profiles_df,
            out_dir,
            subdir=str(cfg.plotting.selected_examples_subdir),
            include_column_index=bool(cfg.plotting.include_column_index),
        )


if __name__ == "__main__":
    main()
