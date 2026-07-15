from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env
from glood.assess_profiles import (
    _ensure_series_axis,
    _iter_example_dirs,
    _normalize_matnum_series,
    _pick_representative_example_ids,
)
from glood.src.assessment.input_materialization import materialized_inference_input


def _safe_relative_l2(pred_values: np.ndarray, true_values: np.ndarray, *, eps: float) -> float:
    pred_values = np.asarray(pred_values, dtype=float)
    true_values = np.asarray(true_values, dtype=float)
    return float(np.linalg.norm(pred_values - true_values) / (np.linalg.norm(true_values) + float(eps)))


def _safe_cosine_similarity(pred_values: np.ndarray, true_values: np.ndarray, *, eps: float) -> float:
    pred_values = np.asarray(pred_values, dtype=float)
    true_values = np.asarray(true_values, dtype=float)
    denom = float(np.linalg.norm(pred_values) * np.linalg.norm(true_values))
    if denom <= float(eps):
        return 1.0 if np.linalg.norm(pred_values - true_values) <= float(eps) else 0.0
    return float(np.dot(pred_values, true_values) / denom)


def _safe_pearson_corr(pred_values: np.ndarray, true_values: np.ndarray, *, eps: float) -> float:
    pred_values = np.asarray(pred_values, dtype=float)
    true_values = np.asarray(true_values, dtype=float)
    pred_centered = pred_values - float(pred_values.mean())
    true_centered = true_values - float(true_values.mean())
    denom = float(np.linalg.norm(pred_centered) * np.linalg.norm(true_centered))
    if denom <= float(eps):
        return 1.0 if np.linalg.norm(pred_centered - true_centered) <= float(eps) else 0.0
    return float(np.dot(pred_centered, true_centered) / denom)


def _optimal_scale(pred_values: np.ndarray, true_values: np.ndarray, *, eps: float) -> float:
    pred_values = np.asarray(pred_values, dtype=float)
    true_values = np.asarray(true_values, dtype=float)
    denom = float(np.dot(pred_values, pred_values))
    if denom <= float(eps):
        return 0.0
    return float(np.dot(pred_values, true_values) / denom)


def _explained_error_fraction(*, raw_error: float, corrected_error: float, eps: float) -> float:
    raw_error = float(raw_error)
    corrected_error = float(corrected_error)
    if raw_error <= float(eps):
        return 0.0 if corrected_error > float(eps) else 1.0
    return float(1.0 - corrected_error / raw_error)


def _structural_residual_ratio(*, raw_error: float, corrected_error: float, eps: float) -> float:
    raw_error = float(raw_error)
    corrected_error = float(corrected_error)
    if raw_error <= float(eps):
        return 0.0 if corrected_error <= float(eps) else float("inf")
    return float(corrected_error / raw_error)


def _extract_field(frame: np.ndarray, *, field_name: str, channels: dict[str, int]) -> np.ndarray:
    if field_name == "p":
        return np.asarray(frame[:, :, channels["p"]], dtype=float)
    if field_name == "ux":
        return np.asarray(frame[:, :, channels["ux"]], dtype=float)
    if field_name == "uy":
        return np.asarray(frame[:, :, channels["uy"]], dtype=float)
    if field_name == "speed":
        ux = np.asarray(frame[:, :, channels["ux"]], dtype=float)
        uy = np.asarray(frame[:, :, channels["uy"]], dtype=float)
        return np.sqrt(ux**2 + uy**2)
    raise ValueError(f"Unsupported field name: {field_name}")


def _overlap_for_shift_2d(
    pred_field: np.ndarray,
    true_field: np.ndarray,
    valid_mask: np.ndarray,
    *,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_field = np.asarray(pred_field, dtype=float)
    true_field = np.asarray(true_field, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    height = int(min(pred_field.shape[0], true_field.shape[0], valid_mask.shape[0]))
    width = int(min(pred_field.shape[1], true_field.shape[1], valid_mask.shape[1]))

    if dy >= 0:
        pred_rows = slice(dy, height)
        true_rows = slice(0, height - dy)
        mask_rows = slice(0, height - dy)
    else:
        k = -int(dy)
        pred_rows = slice(0, height - k)
        true_rows = slice(k, height)
        mask_rows = slice(k, height)

    if dx >= 0:
        pred_cols = slice(dx, width)
        true_cols = slice(0, width - dx)
        mask_cols = slice(0, width - dx)
    else:
        k = -int(dx)
        pred_cols = slice(0, width - k)
        true_cols = slice(k, width)
        mask_cols = slice(k, width)

    pred_overlap = pred_field[pred_rows, pred_cols]
    true_overlap = true_field[true_rows, true_cols]
    mask_overlap = valid_mask[mask_rows, mask_cols]
    if pred_overlap.size == 0 or true_overlap.size == 0 or mask_overlap.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return pred_overlap[mask_overlap], true_overlap[mask_overlap]


def _best_shift_only_2d(
    pred_field: np.ndarray,
    true_field: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_shift_x: int,
    max_shift_y: int,
    eps: float,
    min_overlap_points: int,
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for dy in range(-int(max_shift_y), int(max_shift_y) + 1):
        for dx in range(-int(max_shift_x), int(max_shift_x) + 1):
            pred_values, true_values = _overlap_for_shift_2d(pred_field, true_field, valid_mask, dx=dx, dy=dy)
            if len(pred_values) < int(min_overlap_points):
                continue
            candidate = {
                "best_shift_dx": int(dx),
                "best_shift_dy": int(dy),
                "shift_only_rel_l2": _safe_relative_l2(pred_values, true_values, eps=eps),
                "shift_only_cosine_similarity": _safe_cosine_similarity(pred_values, true_values, eps=eps),
                "shift_only_pearson_corr": _safe_pearson_corr(pred_values, true_values, eps=eps),
                "shift_only_overlap_n": int(len(pred_values)),
            }
            if best is None or float(candidate["shift_only_rel_l2"]) < float(best["shift_only_rel_l2"]):
                best = candidate
    if best is None:
        return {
            "best_shift_dx": 0,
            "best_shift_dy": 0,
            "shift_only_rel_l2": float("nan"),
            "shift_only_cosine_similarity": float("nan"),
            "shift_only_pearson_corr": float("nan"),
            "shift_only_overlap_n": 0,
        }
    return best


def _best_shift_and_scale_2d(
    pred_field: np.ndarray,
    true_field: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_shift_x: int,
    max_shift_y: int,
    eps: float,
    min_overlap_points: int,
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for dy in range(-int(max_shift_y), int(max_shift_y) + 1):
        for dx in range(-int(max_shift_x), int(max_shift_x) + 1):
            pred_values, true_values = _overlap_for_shift_2d(pred_field, true_field, valid_mask, dx=dx, dy=dy)
            if len(pred_values) < int(min_overlap_points):
                continue
            alpha = _optimal_scale(pred_values, true_values, eps=eps)
            scaled_pred = alpha * pred_values
            candidate = {
                "best_shift_scale_dx": int(dx),
                "best_shift_scale_dy": int(dy),
                "best_scale_alpha": float(alpha),
                "shift_scale_rel_l2": _safe_relative_l2(scaled_pred, true_values, eps=eps),
                "shift_scale_cosine_similarity": _safe_cosine_similarity(scaled_pred, true_values, eps=eps),
                "shift_scale_pearson_corr": _safe_pearson_corr(scaled_pred, true_values, eps=eps),
                "shift_scale_overlap_n": int(len(pred_values)),
            }
            if best is None or float(candidate["shift_scale_rel_l2"]) < float(best["shift_scale_rel_l2"]):
                best = candidate
    if best is None:
        return {
            "best_shift_scale_dx": 0,
            "best_shift_scale_dy": 0,
            "best_scale_alpha": 0.0,
            "shift_scale_rel_l2": float("nan"),
            "shift_scale_cosine_similarity": float("nan"),
            "shift_scale_pearson_corr": float("nan"),
            "shift_scale_overlap_n": 0,
        }
    return best


def _build_summary(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    metrics = [
        "raw_rel_l2",
        "scale_only_rel_l2",
        "shift_only_rel_l2",
        "shift_scale_rel_l2",
        "raw_cosine_similarity",
        "scale_only_cosine_similarity",
        "shift_only_cosine_similarity",
        "shift_scale_cosine_similarity",
        "raw_pearson_corr",
        "scale_only_pearson_corr",
        "shift_only_pearson_corr",
        "shift_scale_pearson_corr",
        "best_shift_dx",
        "best_shift_dy",
        "best_shift_scale_dx",
        "best_shift_scale_dy",
        "best_scale_alpha",
        "scale_explained_fraction",
        "shift_explained_fraction",
        "shift_scale_explained_fraction",
        "structural_residual_ratio",
        "true_field_l2",
        "pred_field_l2",
    ]
    agg_map: dict[str, tuple[str, str | callable]] = {
        "n_fields": ("raw_rel_l2", "size"),
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


def _resolve_num_workers(value: object) -> int:
    if isinstance(value, int):
        return max(1, int(value))
    if value is None:
        value = "auto"

    text = str(value).strip().lower()
    if text != "auto":
        return max(1, int(text))

    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        return max(1, int(slurm_cpus_per_task))

    cpu_count = os.cpu_count() or 1
    return max(1, int(cpu_count))


def _process_example_batch(
    *,
    model_name: str,
    geometry: str,
    dataset_label: str,
    example_id: int,
    epoch: int,
    data_dir: str,
    field_names: tuple[str, ...],
    channels: dict[str, int],
    fluid_value: int,
    epsilon: float,
    max_shift_x: int,
    max_shift_y: int,
    min_overlap_points: int,
    is_selected_example: bool,
) -> list[dict[str, float | int | str]]:
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
    fluid_mask = matnum == fluid_value

    rows: list[dict[str, float | int | str]] = []
    for timestep in range(pred.shape[0]):
        valid_mask = fluid_mask[timestep]
        if int(valid_mask.sum()) < int(min_overlap_points):
            continue
        pred_frame = np.asarray(pred[timestep], dtype=float)
        true_frame = np.asarray(true[timestep], dtype=float)

        for field_name in field_names:
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
            scale_explained_fraction = _explained_error_fraction(
                raw_error=raw_rel_l2,
                corrected_error=scale_rel_l2,
                eps=epsilon,
            )
            shift_explained_fraction = _explained_error_fraction(
                raw_error=raw_rel_l2,
                corrected_error=float(shift_only["shift_only_rel_l2"]),
                eps=epsilon,
            )
            shift_scale_explained_fraction = _explained_error_fraction(
                raw_error=raw_rel_l2,
                corrected_error=float(shift_scale["shift_scale_rel_l2"]),
                eps=epsilon,
            )
            structural_residual_ratio = _structural_residual_ratio(
                raw_error=raw_rel_l2,
                corrected_error=float(shift_scale["shift_scale_rel_l2"]),
                eps=epsilon,
            )

            rows.append(
                {
                    "model": model_name,
                    "geometry": geometry,
                    "dataset_label": dataset_label,
                    "example_id": int(example_id),
                    "epoch": int(epoch),
                    "timestep": int(timestep),
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
                    "scale_explained_fraction": float(scale_explained_fraction),
                    "shift_explained_fraction": float(shift_explained_fraction),
                    "shift_scale_explained_fraction": float(shift_scale_explained_fraction),
                    "structural_residual_ratio": float(structural_residual_ratio),
                    "is_selected_example": bool(is_selected_example),
                }
            )
    return rows


def _process_example_batch_star(kwargs: dict[str, object]) -> list[dict[str, float | int | str]]:
    return _process_example_batch(**kwargs)


@hydra.main(version_base=None, config_path="conf", config_name="assess_fieldwise_generic")
def main(cfg: DictConfig) -> None:
    out_dir = env.out()
    with (out_dir / "conf_assess_fieldwise.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    field_names = tuple(str(name) for name in cfg.analysis.fields)
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
    max_shift_x = int(cfg.analysis.max_shift_x)
    max_shift_y = int(cfg.analysis.max_shift_y)
    min_overlap_points = int(cfg.analysis.min_overlap_points)
    num_workers = _resolve_num_workers(getattr(cfg.analysis, "num_workers", "auto"))

    manifest_rows: list[dict[str, int | str]] = []
    rows: list[dict[str, float | int | str]] = []

    for dataset_cfg in cfg.datasets:
        model_name = str(dataset_cfg.model)
        geometry = str(dataset_cfg.geometry)
        dataset_label = str(dataset_cfg.label)
        input_path = Path(str(dataset_cfg.input_dir))
        selected_ids = set(int(x) for x in selected_examples.get(geometry, ()))

        with materialized_inference_input(input_path, label=dataset_label, cfg=cfg) as input_dir:
            example_batches = list(_iter_example_dirs(input_dir))
            if not selected_ids:
                available_example_ids = tuple(sorted({int(example_id) for example_id, _, _ in example_batches}))
                selected_ids = set(_pick_representative_example_ids(available_example_ids, count=selected_examples_count))
            for example_id in sorted(selected_ids):
                manifest_rows.append({"geometry": geometry, "dataset_label": dataset_label, "example_id": example_id})

            worker_kwargs = [
                {
                    "model_name": model_name,
                    "geometry": geometry,
                    "dataset_label": dataset_label,
                    "example_id": int(example_id),
                    "epoch": int(epoch),
                    "data_dir": str(data_dir),
                    "field_names": field_names,
                    "channels": channels,
                    "fluid_value": fluid_value,
                    "epsilon": epsilon,
                    "max_shift_x": max_shift_x,
                    "max_shift_y": max_shift_y,
                    "min_overlap_points": min_overlap_points,
                    "is_selected_example": bool(example_id in selected_ids),
                }
                for example_id, epoch, data_dir in example_batches
            ]

            if num_workers == 1:
                for kwargs in worker_kwargs:
                    rows.extend(_process_example_batch(**kwargs))
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for batch_rows in executor.map(_process_example_batch_star, worker_kwargs):
                        rows.extend(batch_rows)

    if not rows:
        raise ValueError("No fieldwise diagnostics were computed.")

    per_field_df = pd.DataFrame(rows).sort_values(
        ["dataset_label", "field_name", "example_id", "epoch", "timestep"]
    ).reset_index(drop=True)
    overall_df = _build_summary(
        per_field_df,
        group_cols=["model", "geometry", "dataset_label", "field_name"],
    )
    by_timestep_df = _build_summary(
        per_field_df,
        group_cols=["model", "geometry", "dataset_label", "field_name", "timestep"],
    )
    manifest_df = pd.DataFrame(manifest_rows).drop_duplicates().sort_values(
        ["geometry", "dataset_label", "example_id"]
    ).reset_index(drop=True)

    manifest_df.to_csv(out_dir / cfg.outputs.selected_examples_manifest_csv, index=False)
    per_field_df.to_csv(out_dir / cfg.outputs.per_field_csv, index=False)
    overall_df.to_csv(out_dir / cfg.outputs.overall_csv, index=False)
    by_timestep_df.to_csv(out_dir / cfg.outputs.by_timestep_csv, index=False)


if __name__ == "__main__":
    main()
