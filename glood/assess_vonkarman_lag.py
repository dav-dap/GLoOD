from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env
from glood.assess_fieldwise import (
    _best_shift_and_scale_2d,
    _best_shift_only_2d,
    _build_summary as _build_fieldwise_summary,
    _explained_error_fraction,
    _extract_field,
    _optimal_scale,
    _safe_cosine_similarity,
    _safe_pearson_corr,
    _safe_relative_l2,
    _structural_residual_ratio,
)
from glood.assess_profiles import (
    _ensure_series_axis,
    _iter_example_dirs,
    _normalize_matnum_series,
    _relative_l2_error,
    _resolve_section_indices,
)
from glood.src.assessment.input_materialization import materialized_inference_input


def _resolve_num_workers(value: object) -> int:
    if isinstance(value, int):
        return max(1, int(value))
    text = "auto" if value is None else str(value).strip().lower()
    if text != "auto":
        return max(1, int(text))
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        return max(1, int(slurm_cpus_per_task))
    return max(1, int(os.cpu_count() or 1))


def _variant_from_label(label: str) -> str:
    marker = "vonkarmann-"
    if marker in label:
        return label.split(marker, 1)[1]
    return label


def _vorticity(frame: np.ndarray, *, ux_channel: int, uy_channel: int) -> np.ndarray:
    ux = np.asarray(frame[:, :, ux_channel], dtype=float)
    uy = np.asarray(frame[:, :, uy_channel], dtype=float)
    return np.gradient(uy, axis=1) - np.gradient(ux, axis=0)


def _speed(frame: np.ndarray, *, ux_channel: int, uy_channel: int) -> np.ndarray:
    ux = np.asarray(frame[:, :, ux_channel], dtype=float)
    uy = np.asarray(frame[:, :, uy_channel], dtype=float)
    return np.sqrt(ux**2 + uy**2)


def _lag_field(frame: np.ndarray, *, field_name: str, channels: dict[str, int]) -> np.ndarray:
    if field_name == "speed":
        return _speed(frame, ux_channel=channels["ux"], uy_channel=channels["uy"])
    if field_name == "vorticity":
        return _vorticity(frame, ux_channel=channels["ux"], uy_channel=channels["uy"])
    return _extract_field(frame, field_name=field_name, channels=channels)


def _target_timestep(*, pred_timestep: int, alignment: str) -> int | None:
    if alignment == "current":
        return int(pred_timestep)
    if alignment == "previous":
        target_timestep = int(pred_timestep) - 1
        return target_timestep if target_timestep >= 0 else None
    raise ValueError(f"Unsupported alignment: {alignment}")


def _analysis_timesteps(*, series_len: int, start_timestep: int, end_timestep: int | None) -> tuple[int, ...]:
    start = int(max(0, start_timestep))
    stop = int(series_len - 1 if end_timestep is None else min(int(end_timestep), series_len - 1))
    if stop < start:
        return ()
    return tuple(range(start, stop + 1))


def _quantile_summary(df: pd.DataFrame, *, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    agg_map: dict[str, tuple[str, str | Any]] = {
        "n_rows": (metrics[0], "size"),
        "n_examples": ("example_id", "nunique"),
    }
    for metric in metrics:
        agg_map[f"{metric}_median"] = (metric, "median")
        agg_map[f"{metric}_q25"] = (metric, lambda s: float(np.quantile(s, 0.25)))
        agg_map[f"{metric}_q75"] = (metric, lambda s: float(np.quantile(s, 0.75)))
    return (
        df.groupby(group_cols, dropna=False)
        .agg(**agg_map)
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def _lag_alignment_summary(lag_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "current_corr",
        "previous_corr",
        "previous_corr_gain",
        "current_rel_l2",
        "previous_rel_l2",
        "previous_rel_l2_gain",
        "current_scale_alpha",
        "previous_scale_alpha",
    ]
    summary = _quantile_summary(
        lag_df,
        group_cols=["model", "variant", "geometry", "dataset_label", "field_name"],
        metrics=metrics,
    )
    win_df = (
        lag_df.assign(
            previous_corr_better=lambda df: df["previous_corr"] > df["current_corr"],
            previous_l2_better=lambda df: df["previous_rel_l2"] < df["current_rel_l2"],
        )
        .groupby(["model", "variant", "geometry", "dataset_label", "field_name"], dropna=False)
        .agg(
            previous_corr_better_fraction=("previous_corr_better", "mean"),
            previous_l2_better_fraction=("previous_l2_better", "mean"),
        )
        .reset_index()
    )
    return summary.merge(win_df, on=["model", "variant", "geometry", "dataset_label", "field_name"], how="left")


def _profile_summary(per_profile_df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        per_profile_df.groupby(group_cols, dropna=False)
        .agg(
            n_profiles=("profile_error", "size"),
            n_examples=("example_id", "nunique"),
            median_profile_error=("profile_error", "median"),
            q25_profile_error=("profile_error", lambda s: float(np.quantile(s, 0.25))),
            q75_profile_error=("profile_error", lambda s: float(np.quantile(s, 0.75))),
            median_true_profile_l2=("true_profile_l2", "median"),
            median_pred_profile_l2=("pred_profile_l2", "median"),
        )
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    summary["iqr_profile_error"] = summary["q75_profile_error"] - summary["q25_profile_error"]
    return summary


def _process_example(
    *,
    model_name: str,
    variant: str,
    geometry: str,
    dataset_label: str,
    example_id: int,
    epoch: int,
    data_dir: str,
    channels: dict[str, int],
    fluid_value: int,
    epsilon: float,
    lag_fields: tuple[str, ...],
    fieldwise_fields: tuple[str, ...],
    profile_x_fractions: tuple[float, ...],
    alignments: tuple[str, ...],
    start_timestep: int,
    end_timestep: int | None,
    max_shift_x: int,
    max_shift_y: int,
    min_overlap_points: int,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    data_path = Path(data_dir)
    pred = _ensure_series_axis(np.load(data_path / "pred.npy", allow_pickle=False, mmap_mode="r"), name="pred")
    true = _ensure_series_axis(np.load(data_path / "true.npy", allow_pickle=False, mmap_mode="r"), name="true")
    if pred.shape != true.shape:
        raise ValueError(
            f"Pred/true shape mismatch for dataset={dataset_label}, example={example_id}: {pred.shape} vs {true.shape}"
        )
    if pred.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C) arrays, got {pred.shape}")

    matnum = _normalize_matnum_series(
        np.load(data_path / "matnum.npy", allow_pickle=False, mmap_mode="r"),
        series_len=pred.shape[0],
    )
    fluid_mask = matnum == int(fluid_value)
    sections = _resolve_section_indices(pred.shape[2], profile_x_fractions)

    lag_rows: list[dict[str, float | int | str]] = []
    fieldwise_rows: list[dict[str, float | int | str]] = []
    profile_rows: list[dict[str, float | int | str]] = []
    timesteps = _analysis_timesteps(
        series_len=pred.shape[0],
        start_timestep=start_timestep,
        end_timestep=end_timestep,
    )

    for pred_timestep in timesteps:
        previous_timestep = pred_timestep - 1
        if previous_timestep >= 0:
            pred_frame = np.asarray(pred[pred_timestep], dtype=float)
            true_current_frame = np.asarray(true[pred_timestep], dtype=float)
            true_previous_frame = np.asarray(true[previous_timestep], dtype=float)
            current_mask = fluid_mask[pred_timestep]
            previous_mask = fluid_mask[previous_timestep]
            common_mask = current_mask & previous_mask
            if int(common_mask.sum()) >= int(min_overlap_points):
                for field_name in lag_fields:
                    pred_field = _lag_field(pred_frame, field_name=field_name, channels=channels)
                    current_field = _lag_field(true_current_frame, field_name=field_name, channels=channels)
                    previous_field = _lag_field(true_previous_frame, field_name=field_name, channels=channels)
                    pred_values = pred_field[common_mask]
                    current_values = current_field[common_mask]
                    previous_values = previous_field[common_mask]

                    current_corr = _safe_pearson_corr(pred_values, current_values, eps=epsilon)
                    previous_corr = _safe_pearson_corr(pred_values, previous_values, eps=epsilon)
                    current_rel_l2 = _safe_relative_l2(pred_values, current_values, eps=epsilon)
                    previous_rel_l2 = _safe_relative_l2(pred_values, previous_values, eps=epsilon)
                    lag_rows.append(
                        {
                            "model": model_name,
                            "variant": variant,
                            "geometry": geometry,
                            "dataset_label": dataset_label,
                            "example_id": int(example_id),
                            "epoch": int(epoch),
                            "pred_timestep": int(pred_timestep),
                            "current_target_timestep": int(pred_timestep),
                            "previous_target_timestep": int(previous_timestep),
                            "field_name": field_name,
                            "n_valid_points": int(common_mask.sum()),
                            "current_corr": float(current_corr),
                            "previous_corr": float(previous_corr),
                            "previous_corr_gain": float(previous_corr - current_corr),
                            "current_rel_l2": float(current_rel_l2),
                            "previous_rel_l2": float(previous_rel_l2),
                            "previous_rel_l2_gain": float(current_rel_l2 - previous_rel_l2),
                            "current_scale_alpha": float(_optimal_scale(pred_values, current_values, eps=epsilon)),
                            "previous_scale_alpha": float(_optimal_scale(pred_values, previous_values, eps=epsilon)),
                            "pred_field_l2": float(np.linalg.norm(pred_values)),
                            "current_target_l2": float(np.linalg.norm(current_values)),
                            "previous_target_l2": float(np.linalg.norm(previous_values)),
                        }
                    )

        for alignment in alignments:
            target_timestep = _target_timestep(pred_timestep=pred_timestep, alignment=alignment)
            if target_timestep is None:
                continue
            pred_frame = np.asarray(pred[pred_timestep], dtype=float)
            true_frame = np.asarray(true[target_timestep], dtype=float)
            valid_mask = fluid_mask[target_timestep]
            if int(valid_mask.sum()) < int(min_overlap_points):
                continue

            for field_name in fieldwise_fields:
                pred_field = _extract_field(pred_frame, field_name=field_name, channels=channels)
                true_field = _extract_field(true_frame, field_name=field_name, channels=channels)
                pred_values = pred_field[valid_mask]
                true_values = true_field[valid_mask]

                raw_rel_l2 = _safe_relative_l2(pred_values, true_values, eps=epsilon)
                raw_cos = _safe_cosine_similarity(pred_values, true_values, eps=epsilon)
                raw_corr = _safe_pearson_corr(pred_values, true_values, eps=epsilon)

                scale_alpha = _optimal_scale(pred_values, true_values, eps=epsilon)
                scaled_pred_values = scale_alpha * pred_values
                scale_rel_l2 = _safe_relative_l2(scaled_pred_values, true_values, eps=epsilon)
                scale_cos = _safe_cosine_similarity(scaled_pred_values, true_values, eps=epsilon)
                scale_corr = _safe_pearson_corr(scaled_pred_values, true_values, eps=epsilon)

                shift_only = _best_shift_only_2d(
                    pred_field,
                    true_field,
                    valid_mask,
                    max_shift_x=max_shift_x,
                    max_shift_y=max_shift_y,
                    eps=epsilon,
                    min_overlap_points=min_overlap_points,
                )
                shift_scale = _best_shift_and_scale_2d(
                    pred_field,
                    true_field,
                    valid_mask,
                    max_shift_x=max_shift_x,
                    max_shift_y=max_shift_y,
                    eps=epsilon,
                    min_overlap_points=min_overlap_points,
                )
                fieldwise_rows.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "geometry": geometry,
                        "dataset_label": dataset_label,
                        "alignment": alignment,
                        "example_id": int(example_id),
                        "epoch": int(epoch),
                        "pred_timestep": int(pred_timestep),
                        "target_timestep": int(target_timestep),
                        "field_name": field_name,
                        "n_valid_points": int(valid_mask.sum()),
                        "true_field_l2": float(np.linalg.norm(true_values)),
                        "pred_field_l2": float(np.linalg.norm(pred_values)),
                        "raw_rel_l2": float(raw_rel_l2),
                        "raw_cosine_similarity": float(raw_cos),
                        "raw_pearson_corr": float(raw_corr),
                        "scale_only_alpha": float(scale_alpha),
                        "scale_only_rel_l2": float(scale_rel_l2),
                        "scale_only_cosine_similarity": float(scale_cos),
                        "scale_only_pearson_corr": float(scale_corr),
                        **shift_only,
                        **shift_scale,
                        "scale_explained_fraction": float(
                            _explained_error_fraction(raw_error=raw_rel_l2, corrected_error=scale_rel_l2, eps=epsilon)
                        ),
                        "shift_explained_fraction": float(
                            _explained_error_fraction(
                                raw_error=raw_rel_l2,
                                corrected_error=float(shift_only["shift_only_rel_l2"]),
                                eps=epsilon,
                            )
                        ),
                        "shift_scale_explained_fraction": float(
                            _explained_error_fraction(
                                raw_error=raw_rel_l2,
                                corrected_error=float(shift_scale["shift_scale_rel_l2"]),
                                eps=epsilon,
                            )
                        ),
                        "structural_residual_ratio": float(
                            _structural_residual_ratio(
                                raw_error=raw_rel_l2,
                                corrected_error=float(shift_scale["shift_scale_rel_l2"]),
                                eps=epsilon,
                            )
                        ),
                    }
                )

            pred_ux = pred_frame[:, :, channels["ux"]]
            true_ux = true_frame[:, :, channels["ux"]]
            for x_fraction, x_index in sections:
                profile_mask = valid_mask[:, x_index]
                n_valid_points = int(profile_mask.sum())
                if n_valid_points == 0:
                    continue
                pred_profile = pred_ux[:, x_index][profile_mask]
                true_profile = true_ux[:, x_index][profile_mask]
                profile_rows.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "geometry": geometry,
                        "dataset_label": dataset_label,
                        "alignment": alignment,
                        "example_id": int(example_id),
                        "epoch": int(epoch),
                        "pred_timestep": int(pred_timestep),
                        "target_timestep": int(target_timestep),
                        "x_fraction": float(x_fraction),
                        "x_index": int(x_index),
                        "n_valid_points": n_valid_points,
                        "true_profile_l2": float(np.linalg.norm(true_profile)),
                        "pred_profile_l2": float(np.linalg.norm(pred_profile)),
                        "profile_error": _relative_l2_error(pred_profile, true_profile, eps=epsilon),
                    }
                )

    return lag_rows, fieldwise_rows, profile_rows


def _process_example_star(kwargs: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    return _process_example(**kwargs)


@hydra.main(version_base=None, config_path="conf", config_name="assess_vonkarman_lag_generic")
def main(cfg: DictConfig) -> None:
    out_dir = env.out()
    with (out_dir / "conf_assess_vonkarman_lag.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    channels = {
        "p": int(cfg.analysis.p_channel),
        "ux": int(cfg.analysis.ux_channel),
        "uy": int(cfg.analysis.uy_channel),
    }
    fluid_value = int(cfg.analysis.fluid_value)
    epsilon = float(cfg.analysis.error_epsilon)
    lag_fields = tuple(str(x) for x in cfg.analysis.lag_fields)
    fieldwise_fields = tuple(str(x) for x in cfg.analysis.fieldwise_fields)
    profile_x_fractions = tuple(float(x) for x in cfg.analysis.profile_x_fractions)
    alignments = tuple(str(x) for x in cfg.analysis.alignments)
    start_timestep = int(cfg.analysis.start_timestep)
    end_timestep_raw = getattr(cfg.analysis, "end_timestep", None)
    end_timestep = None if end_timestep_raw is None else int(end_timestep_raw)
    max_shift_x = int(cfg.analysis.max_shift_x)
    max_shift_y = int(cfg.analysis.max_shift_y)
    min_overlap_points = int(cfg.analysis.min_overlap_points)
    max_examples = int(getattr(cfg.analysis, "max_examples", 0))
    num_workers = _resolve_num_workers(getattr(cfg.analysis, "num_workers", "auto"))

    lag_rows: list[dict[str, object]] = []
    fieldwise_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    for dataset_cfg in cfg.datasets:
        model_name = str(dataset_cfg.model)
        geometry = str(dataset_cfg.geometry)
        dataset_label = str(dataset_cfg.label)
        variant = str(getattr(dataset_cfg, "variant", _variant_from_label(dataset_label)))
        input_path = Path(str(dataset_cfg.input_dir))

        with materialized_inference_input(input_path, label=dataset_label, cfg=cfg) as input_dir:
            example_batches = list(_iter_example_dirs(input_dir))
            if max_examples > 0:
                example_batches = example_batches[:max_examples]
            print(f"{dataset_label}: queued {len(example_batches)} examples from {input_path}")

            worker_kwargs = [
                {
                    "model_name": model_name,
                    "variant": variant,
                    "geometry": geometry,
                    "dataset_label": dataset_label,
                    "example_id": int(example_id),
                    "epoch": int(epoch),
                    "data_dir": str(data_dir),
                    "channels": channels,
                    "fluid_value": fluid_value,
                    "epsilon": epsilon,
                    "lag_fields": lag_fields,
                    "fieldwise_fields": fieldwise_fields,
                    "profile_x_fractions": profile_x_fractions,
                    "alignments": alignments,
                    "start_timestep": start_timestep,
                    "end_timestep": end_timestep,
                    "max_shift_x": max_shift_x,
                    "max_shift_y": max_shift_y,
                    "min_overlap_points": min_overlap_points,
                }
                for example_id, epoch, data_dir in example_batches
            ]

            if num_workers == 1:
                for kwargs in worker_kwargs:
                    batch_lag, batch_fieldwise, batch_profile = _process_example(**kwargs)
                    lag_rows.extend(batch_lag)
                    fieldwise_rows.extend(batch_fieldwise)
                    profile_rows.extend(batch_profile)
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for batch_lag, batch_fieldwise, batch_profile in executor.map(_process_example_star, worker_kwargs):
                        lag_rows.extend(batch_lag)
                        fieldwise_rows.extend(batch_fieldwise)
                        profile_rows.extend(batch_profile)

    if not lag_rows:
        raise ValueError("No lag-alignment rows were computed.")
    if not fieldwise_rows:
        raise ValueError("No fieldwise rows were computed.")
    if not profile_rows:
        raise ValueError("No profile rows were computed.")

    lag_df = pd.DataFrame(lag_rows).sort_values(
        ["dataset_label", "field_name", "example_id", "pred_timestep"]
    ).reset_index(drop=True)
    lag_summary_df = _lag_alignment_summary(lag_df)
    lag_by_timestep_df = _quantile_summary(
        lag_df,
        group_cols=["model", "variant", "geometry", "dataset_label", "field_name", "pred_timestep"],
        metrics=[
            "current_corr",
            "previous_corr",
            "previous_corr_gain",
            "current_rel_l2",
            "previous_rel_l2",
            "previous_rel_l2_gain",
        ],
    )

    fieldwise_df = pd.DataFrame(fieldwise_rows).sort_values(
        ["dataset_label", "alignment", "field_name", "example_id", "pred_timestep"]
    ).reset_index(drop=True)
    fieldwise_overall_df = _build_fieldwise_summary(
        fieldwise_df,
        group_cols=["model", "variant", "geometry", "dataset_label", "alignment", "field_name"],
    )
    fieldwise_by_timestep_df = _build_fieldwise_summary(
        fieldwise_df,
        group_cols=[
            "model",
            "variant",
            "geometry",
            "dataset_label",
            "alignment",
            "field_name",
            "pred_timestep",
        ],
    )

    profile_df = pd.DataFrame(profile_rows).sort_values(
        ["dataset_label", "alignment", "example_id", "pred_timestep", "x_fraction"]
    ).reset_index(drop=True)
    profile_summary_df = _profile_summary(
        profile_df,
        group_cols=[
            "model",
            "variant",
            "geometry",
            "dataset_label",
            "alignment",
            "x_fraction",
            "x_index",
        ],
    )
    profile_overall_df = _profile_summary(
        profile_df,
        group_cols=["model", "variant", "geometry", "dataset_label", "alignment"],
    )
    profile_by_timestep_df = _profile_summary(
        profile_df,
        group_cols=[
            "model",
            "variant",
            "geometry",
            "dataset_label",
            "alignment",
            "pred_timestep",
        ],
    )

    lag_df.to_csv(out_dir / cfg.outputs.lag_per_frame_csv, index=False)
    lag_summary_df.to_csv(out_dir / cfg.outputs.lag_summary_csv, index=False)
    lag_by_timestep_df.to_csv(out_dir / cfg.outputs.lag_by_timestep_csv, index=False)
    fieldwise_df.to_csv(out_dir / cfg.outputs.fieldwise_per_field_csv, index=False)
    fieldwise_overall_df.to_csv(out_dir / cfg.outputs.fieldwise_overall_csv, index=False)
    fieldwise_by_timestep_df.to_csv(out_dir / cfg.outputs.fieldwise_by_timestep_csv, index=False)
    profile_df.to_csv(out_dir / cfg.outputs.profile_per_profile_csv, index=False)
    profile_summary_df.to_csv(out_dir / cfg.outputs.profile_summary_csv, index=False)
    profile_overall_df.to_csv(out_dir / cfg.outputs.profile_overall_csv, index=False)
    profile_by_timestep_df.to_csv(out_dir / cfg.outputs.profile_by_timestep_csv, index=False)

    print(f"Wrote Von Karman lag assessment outputs to {out_dir}")


if __name__ == "__main__":
    main()
