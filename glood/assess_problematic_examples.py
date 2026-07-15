from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

import glood.src.utils.env as env

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARSER_TO_DATASET_TARGET = {
    "glood.src.datasets.parsing.ParseSeries": "glood.src.datasets.sharded_dataset.ShardedDatasetSeries",
    "glood.src.datasets.parsing.ParseSeriesWithMatnum": "glood.src.datasets.sharded_dataset.ShardedDatasetSeriesWithMatnum",
    "glood.src.datasets.parsing.ParseSingle": "glood.src.datasets.sharded_dataset.ShardedDatasetSingle",
}
FALLBACK_PARSER_TARGET = "glood.src.datasets.parsing.ParseSeries"


def _label_from_shard_dir(shard_dir: Path) -> str:
    candidate = Path(shard_dir)
    if candidate.name == "dataset":
        candidate = candidate.parent
    if candidate.name.isdigit():
        candidate = candidate.parent
    return candidate.name


def _find_dataset_hydra_config(shard_dir: Path) -> Path | None:
    for level, candidate in enumerate((shard_dir, *shard_dir.parents)):
        if level > 4:
            break
        config_path = candidate / ".hydra" / "config.yaml"
        if config_path.is_file():
            return config_path
    return None


def _dataset_target_from_parsing(parser_target: str) -> str:
    dataset_target = PARSER_TO_DATASET_TARGET.get(parser_target)
    if dataset_target is None:
        raise ValueError(f"Unsupported dataset parsing target for problematic example audit: {parser_target}")
    return dataset_target


def hack_conf(cfg: DictConfig) -> DictConfig:
    source_cfg = cfg.get("problematic_input_source")
    if source_cfg is None:
        return cfg
    if source_cfg.get("_target_") != "glood.src.assessment.problematic_sources.ShardedDatasetSource":
        return cfg

    shard_dir = Path(str(cfg.problematic_input.dataset.shard_dir))
    dataset_config_path = _find_dataset_hydra_config(shard_dir)
    if dataset_config_path is None:
        parser_target = FALLBACK_PARSER_TARGET
    else:
        dataset_cfg = OmegaConf.load(dataset_config_path)
        parser_target = str(dataset_cfg.parsing._target_)

    dataset_target = _dataset_target_from_parsing(parser_target)
    with open_dict(cfg):
        cfg.problematic_input_source.dataset._target_ = dataset_target
        cfg.problematic_input_source.dataset_label = _label_from_shard_dir(shard_dir)
        cfg.problematic_input_source.dataset_config_path = (
            None if dataset_config_path is None else str(dataset_config_path)
        )
        cfg.problematic_input_source.parsing_target = parser_target
    return cfg


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


def _nanmasked(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=float).copy()
    data[~mask] = np.nan
    return data


def _masked_quantile(values: np.ndarray, *, q: float, fallback: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(fallback)
    return float(np.quantile(finite, q))


def _gradient_magnitude(field: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(field, dtype=float))
    return np.sqrt(gx**2 + gy**2)


def _laplacian_abs(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    gy, gx = np.gradient(field)
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    return np.abs(gxx + gyy)


def _field_limits(masked_field: np.ndarray, *, symmetric: bool) -> tuple[float, float]:
    lo = _masked_quantile(masked_field, q=0.02)
    hi = _masked_quantile(masked_field, q=0.98)
    if symmetric:
        bound = max(abs(lo), abs(hi), 1.0e-8)
        return -bound, bound
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        finite_mean = np.nanmean(masked_field)
        center = float(finite_mean) if np.isfinite(finite_mean) else 0.0
        return center - 1.0, center + 1.0
    return lo, hi


def _fluid_mask_for_snapshot(snapshot, *, fluid_value: int) -> np.ndarray:
    frame = np.asarray(snapshot.frame)
    if snapshot.matnum is None:
        return np.ones(frame.shape[:2], dtype=bool)
    return np.asarray(snapshot.matnum) == int(fluid_value)


def _plot_snapshot(
    *,
    out_path: Path,
    dataset_label: str,
    review_index: int,
    issue_label: str,
    example_id: int,
    timestep: int,
    source_frame: np.ndarray,
    fluid_mask: np.ndarray,
    channels: dict[str, int],
    source_name: str,
    gradient_field: str,
    oscillation_field: str,
    row: pd.Series,
) -> None:
    p_field = _extract_field(source_frame, field_name="p", channels=channels)
    ux_field = _extract_field(source_frame, field_name="ux", channels=channels)
    uy_field = _extract_field(source_frame, field_name="uy", channels=channels)
    speed_field = _extract_field(source_frame, field_name="speed", channels=channels)

    gradient_target = _extract_field(source_frame, field_name=gradient_field, channels=channels)
    oscillation_target = _extract_field(source_frame, field_name=oscillation_field, channels=channels)
    gradient_map = _gradient_magnitude(gradient_target)
    oscillation_map = _laplacian_abs(oscillation_target)

    masked_p = _nanmasked(p_field, fluid_mask)
    masked_ux = _nanmasked(ux_field, fluid_mask)
    masked_uy = _nanmasked(uy_field, fluid_mask)
    masked_speed = _nanmasked(speed_field, fluid_mask)
    masked_grad = _nanmasked(gradient_map, fluid_mask)
    masked_osc = _nanmasked(oscillation_map, fluid_mask)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("pressure", masked_p, "coolwarm", _field_limits(masked_p, symmetric=True)),
        ("ux", masked_ux, "coolwarm", _field_limits(masked_ux, symmetric=True)),
        ("uy", masked_uy, "coolwarm", _field_limits(masked_uy, symmetric=True)),
        ("speed", masked_speed, "viridis", _field_limits(masked_speed, symmetric=False)),
        (f"|grad {gradient_field}|", masked_grad, "magma", _field_limits(masked_grad, symmetric=False)),
        (f"|lap {oscillation_field}|", masked_osc, "magma", _field_limits(masked_osc, symmetric=False)),
    ]

    for ax, (title, image, cmap, (vmin, vmax)) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        " | ".join(
            [
                f"#{review_index:04d}",
                issue_label,
                dataset_label,
                f"example={example_id}",
                f"t={timestep}",
                f"source={source_name}",
                f"grad={row['gradient_score']:.4g}",
                f"osc={row['oscillation_score']:.4g}",
            ]
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _rank_selected_snapshots(
    metrics_df: pd.DataFrame,
    *,
    top_k_gradient: int,
    top_k_oscillation: int,
    min_gradient_score: float,
    min_oscillation_score: float,
) -> pd.DataFrame:
    grad_pool = metrics_df[metrics_df["gradient_score"] > float(min_gradient_score)].copy()
    grad_top = grad_pool.nlargest(top_k_gradient, columns=["gradient_score", "oscillation_score"]).copy()
    grad_top["selected_by_gradient"] = True
    grad_top["selected_by_oscillation"] = False

    osc_pool = metrics_df[metrics_df["oscillation_score"] > float(min_oscillation_score)].copy()
    osc_top = osc_pool.nlargest(top_k_oscillation, columns=["oscillation_score", "gradient_score"]).copy()
    osc_top["selected_by_gradient"] = False
    osc_top["selected_by_oscillation"] = True

    merged = pd.concat([grad_top, osc_top], ignore_index=True)
    if merged.empty:
        raise ValueError("No problematic snapshots passed the configured gradient/oscillation thresholds")
    merged = (
        merged.groupby(["dataset_label", "source_name", "example_id", "epoch", "timestep"], dropna=False)
        .agg(
            selected_by_gradient=("selected_by_gradient", "max"),
            selected_by_oscillation=("selected_by_oscillation", "max"),
            gradient_score=("gradient_score", "first"),
            oscillation_score=("oscillation_score", "first"),
            dominant_gradient_field=("dominant_gradient_field", "first"),
            dominant_oscillation_field=("dominant_oscillation_field", "first"),
            n_valid_points=("n_valid_points", "first"),
            source_field_l2=("source_field_l2", "first"),
            speed_ref_snapshot=("speed_ref_snapshot", "first"),
            speed_ref=("speed_ref", "first"),
            speed_ref_eps=("speed_ref_eps", "first"),
            gradient_score_raw=("gradient_score_raw", "first"),
            gradient_score_normalized=("gradient_score_normalized", "first"),
            gradient_normalized_by_speed=("gradient_normalized_by_speed", "first"),
            **{
                c: (c, "first")
                for c in merged.columns
                if c.endswith("_grad_q99")
                or c.endswith("_grad_q95")
                or c.endswith("_osc_q99")
                or c.endswith("_osc_q95")
                or c.endswith("_grad_score_raw")
                or c.endswith("_grad_score_normalized")
                or c.endswith("_grad_score_quantile")
                or c.endswith("_osc_score_quantile")
            },
        )
        .reset_index()
    )

    merged["issue_label"] = np.where(
        merged["selected_by_gradient"] & merged["selected_by_oscillation"],
        "gradient+oscillation",
        np.where(merged["selected_by_gradient"], "gradient", "oscillation"),
    )
    merged["priority_score"] = np.maximum(
        merged["gradient_score"].to_numpy(dtype=float),
        merged["oscillation_score"].to_numpy(dtype=float),
    )
    merged = merged.sort_values(
        ["priority_score", "gradient_score", "oscillation_score", "example_id", "timestep"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    merged["review_index"] = np.arange(1, len(merged) + 1, dtype=int)
    return merged


def _example_key_columns() -> list[str]:
    return ["dataset_label", "source_name", "example_id", "epoch"]


def _sort_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["priority_score", "gradient_score", "oscillation_score", "example_id", "timestep"]
    ascending = [False, False, False, True, True]
    return df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def _rank_selected_examples(
    metrics_df: pd.DataFrame,
    *,
    top_k_gradient: int,
    top_k_oscillation: int,
    min_gradient_score: float,
    min_oscillation_score: float,
    snapshots_per_example: int,
) -> pd.DataFrame:
    if snapshots_per_example <= 0:
        raise ValueError(f"snapshots_per_example must be positive, got {snapshots_per_example}")

    example_keys = _example_key_columns()
    grad_pool = metrics_df[metrics_df["gradient_score"] > float(min_gradient_score)].copy()
    if grad_pool.empty:
        grad_best = grad_pool
    else:
        grad_best = grad_pool.loc[grad_pool.groupby(example_keys, dropna=False)["gradient_score"].idxmax()].copy()
        grad_best = grad_best.nlargest(top_k_gradient, columns=["gradient_score", "oscillation_score"]).copy()
    grad_best["selected_by_gradient"] = True
    grad_best["selected_by_oscillation"] = False

    osc_pool = metrics_df[metrics_df["oscillation_score"] > float(min_oscillation_score)].copy()
    if osc_pool.empty:
        osc_best = osc_pool
    else:
        osc_best = osc_pool.loc[osc_pool.groupby(example_keys, dropna=False)["oscillation_score"].idxmax()].copy()
        osc_best = osc_best.nlargest(top_k_oscillation, columns=["oscillation_score", "gradient_score"]).copy()
    osc_best["selected_by_gradient"] = False
    osc_best["selected_by_oscillation"] = True

    selected_examples = pd.concat([grad_best, osc_best], ignore_index=True)
    if selected_examples.empty:
        raise ValueError("No problematic examples passed the configured gradient/oscillation thresholds")

    selected_examples["example_priority_score"] = np.maximum(
        selected_examples["gradient_score"].to_numpy(dtype=float),
        selected_examples["oscillation_score"].to_numpy(dtype=float),
    )
    selected_examples = (
        selected_examples.groupby(example_keys, dropna=False)
        .agg(
            selected_by_gradient=("selected_by_gradient", "max"),
            selected_by_oscillation=("selected_by_oscillation", "max"),
            example_priority_score=("example_priority_score", "max"),
        )
        .reset_index()
    )

    selected_lookup = {
        tuple(row[col] for col in example_keys): row
        for _, row in selected_examples.iterrows()
    }
    selected_key_set = set(selected_lookup)

    representative_rows: list[pd.DataFrame] = []
    for key, example_df in metrics_df.groupby(example_keys, dropna=False, sort=False):
        if key not in selected_key_set:
            continue
        flags = selected_lookup[key]
        candidate_parts: list[pd.DataFrame] = []
        if bool(flags["selected_by_gradient"]):
            candidate_parts.append(example_df.nlargest(1, columns=["gradient_score", "oscillation_score"]))
        if bool(flags["selected_by_oscillation"]):
            candidate_parts.append(example_df.nlargest(1, columns=["oscillation_score", "gradient_score"]))

        candidates = pd.concat(candidate_parts, ignore_index=False) if candidate_parts else example_df.head(0)
        candidates = candidates.drop_duplicates(subset=example_keys + ["timestep"])
        if len(candidates) < snapshots_per_example:
            remaining = example_df.drop(index=candidates.index, errors="ignore").copy()
            remaining["priority_score"] = np.maximum(
                remaining["gradient_score"].to_numpy(dtype=float),
                remaining["oscillation_score"].to_numpy(dtype=float),
            )
            candidates = pd.concat(
                [
                    candidates,
                    _sort_by_priority(remaining).head(snapshots_per_example - len(candidates)),
                ],
                ignore_index=False,
            )

        candidates = candidates.copy()
        candidates["selected_by_gradient"] = bool(flags["selected_by_gradient"])
        candidates["selected_by_oscillation"] = bool(flags["selected_by_oscillation"])
        candidates["example_priority_score"] = float(flags["example_priority_score"])
        candidates["priority_score"] = np.maximum(
            candidates["gradient_score"].to_numpy(dtype=float),
            candidates["oscillation_score"].to_numpy(dtype=float),
        )
        candidates = _sort_by_priority(candidates).head(snapshots_per_example)
        representative_rows.append(candidates)

    if not representative_rows:
        raise ValueError("No representative snapshots found for selected problematic examples")

    merged = pd.concat(representative_rows, ignore_index=True)
    merged["issue_label"] = np.where(
        merged["selected_by_gradient"] & merged["selected_by_oscillation"],
        "gradient+oscillation",
        np.where(merged["selected_by_gradient"], "gradient", "oscillation"),
    )
    merged = merged.sort_values(
        ["example_priority_score", "priority_score", "gradient_score", "oscillation_score", "example_id", "timestep"],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)
    merged["review_index"] = np.arange(1, len(merged) + 1, dtype=int)
    return merged


def _rank_selected(
    metrics_df: pd.DataFrame,
    *,
    grouping: str,
    top_k_gradient: int,
    top_k_oscillation: int,
    min_gradient_score: float,
    min_oscillation_score: float,
    snapshots_per_example: int,
) -> pd.DataFrame:
    grouping = str(grouping).strip().lower()
    if grouping == "snapshot":
        return _rank_selected_snapshots(
            metrics_df,
            top_k_gradient=top_k_gradient,
            top_k_oscillation=top_k_oscillation,
            min_gradient_score=min_gradient_score,
            min_oscillation_score=min_oscillation_score,
        )
    if grouping == "example":
        return _rank_selected_examples(
            metrics_df,
            top_k_gradient=top_k_gradient,
            top_k_oscillation=top_k_oscillation,
            min_gradient_score=min_gradient_score,
            min_oscillation_score=min_oscillation_score,
            snapshots_per_example=snapshots_per_example,
        )
    raise ValueError("selection.grouping must be either 'snapshot' or 'example'.")


def _collect_metrics(
    input_source,
    *,
    audit_fields: tuple[str, ...],
    channels: dict[str, int],
    fluid_value: int,
    min_valid_points: int,
    score_quantile: float,
    normalize_gradient_by_example_speed: bool,
    speed_reference_quantile: float,
    speed_reference_eps: float,
) -> pd.DataFrame:
    if not 0.0 <= float(score_quantile) <= 1.0:
        raise ValueError(f"score_quantile must be in [0, 1], got {score_quantile}")
    if not 0.0 <= float(speed_reference_quantile) <= 1.0:
        raise ValueError(f"speed_reference_quantile must be in [0, 1], got {speed_reference_quantile}")
    rows: list[dict[str, float | int | str]] = []
    for snapshot in input_source.iter_snapshots():
        frame = np.asarray(snapshot.frame, dtype=float)
        fluid_mask = _fluid_mask_for_snapshot(snapshot, fluid_value=fluid_value)
        n_valid = int(fluid_mask.sum())
        if n_valid < min_valid_points:
            continue

        row: dict[str, float | int | str] = {
            "dataset_label": str(snapshot.dataset_label),
            "source_name": str(snapshot.source_name),
            "example_id": int(snapshot.example_id),
            "epoch": int(snapshot.epoch),
            "timestep": int(snapshot.timestep),
            "n_valid_points": n_valid,
        }
        speed_values = _extract_field(frame, field_name="speed", channels=channels)[fluid_mask]
        row["speed_ref_snapshot"] = _masked_quantile(speed_values, q=float(speed_reference_quantile))
        for field_name in audit_fields:
            field = _extract_field(frame, field_name=field_name, channels=channels)
            masked_values = field[fluid_mask]
            grad_map = _gradient_magnitude(field)[fluid_mask]
            osc_map = _laplacian_abs(field)[fluid_mask]
            row[f"{field_name}_field_l2"] = float(np.linalg.norm(masked_values))
            row[f"{field_name}_grad_q95"] = _masked_quantile(grad_map, q=0.95)
            row[f"{field_name}_grad_q99"] = _masked_quantile(grad_map, q=0.99)
            row[f"{field_name}_osc_q95"] = _masked_quantile(osc_map, q=0.95)
            row[f"{field_name}_osc_q99"] = _masked_quantile(osc_map, q=0.99)
            row[f"{field_name}_grad_score_raw"] = _masked_quantile(grad_map, q=float(score_quantile))
            row[f"{field_name}_osc_score_quantile"] = _masked_quantile(osc_map, q=float(score_quantile))

        row["source_field_l2"] = float(
            np.linalg.norm(_extract_field(frame, field_name="speed", channels=channels)[fluid_mask])
        )
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise ValueError("No plottable snapshots found for the configured problematic_input_source")

    example_keys = _example_key_columns()
    metrics_df["speed_ref"] = metrics_df.groupby(example_keys, dropna=False)["speed_ref_snapshot"].transform("max")
    metrics_df["speed_ref_eps"] = float(speed_reference_eps)
    denominator = metrics_df["speed_ref"].to_numpy(dtype=float) + float(speed_reference_eps)
    for field_name in audit_fields:
        raw_col = f"{field_name}_grad_score_raw"
        normalized_col = f"{field_name}_grad_score_normalized"
        final_col = f"{field_name}_grad_score_quantile"
        metrics_df[normalized_col] = metrics_df[raw_col].to_numpy(dtype=float) / denominator
        metrics_df[final_col] = metrics_df[normalized_col] if normalize_gradient_by_example_speed else metrics_df[raw_col]

    grad_score_cols = [f"{field_name}_grad_score_quantile" for field_name in audit_fields]
    osc_score_cols = [f"{field_name}_osc_score_quantile" for field_name in audit_fields]
    grad_raw_cols = [f"{field_name}_grad_score_raw" for field_name in audit_fields]
    grad_norm_cols = [f"{field_name}_grad_score_normalized" for field_name in audit_fields]

    dominant_grad_idx = metrics_df[grad_score_cols].to_numpy(dtype=float).argmax(axis=1)
    dominant_osc_idx = metrics_df[osc_score_cols].to_numpy(dtype=float).argmax(axis=1)
    audit_fields_array = np.asarray(audit_fields, dtype=object)
    metrics_df["dominant_gradient_field"] = audit_fields_array[dominant_grad_idx]
    metrics_df["dominant_oscillation_field"] = audit_fields_array[dominant_osc_idx]
    metrics_df["gradient_score"] = metrics_df[grad_score_cols].to_numpy(dtype=float).max(axis=1)
    metrics_df["oscillation_score"] = metrics_df[osc_score_cols].to_numpy(dtype=float).max(axis=1)
    metrics_df["gradient_score_raw"] = metrics_df[grad_raw_cols].to_numpy(dtype=float)[
        np.arange(len(metrics_df)), dominant_grad_idx
    ]
    metrics_df["gradient_score_normalized"] = metrics_df[grad_norm_cols].to_numpy(dtype=float)[
        np.arange(len(metrics_df)), dominant_grad_idx
    ]
    metrics_df["gradient_normalized_by_speed"] = bool(normalize_gradient_by_example_speed)

    return metrics_df.sort_values(["gradient_score", "oscillation_score"], ascending=[False, False]).reset_index(drop=True)


def _plot_selected_snapshots(input_source, selected_df: pd.DataFrame, *, plot_dir: Path, channels: dict[str, int], fluid_value: int) -> None:
    selected_lookup = {
        (
            str(row.dataset_label),
            str(row.source_name),
            int(row.example_id),
            int(row.epoch),
            int(row.timestep),
        ): pd.Series(row._asdict())
        for row in selected_df.itertuples(index=False)
    }

    for snapshot in input_source.iter_snapshots():
        key = (
            str(snapshot.dataset_label),
            str(snapshot.source_name),
            int(snapshot.example_id),
            int(snapshot.epoch),
            int(snapshot.timestep),
        )
        row = selected_lookup.get(key)
        if row is None:
            continue
        frame = np.asarray(snapshot.frame, dtype=float)
        fluid_mask = _fluid_mask_for_snapshot(snapshot, fluid_value=fluid_value)
        stem = plot_dir / (
            f"{int(row['review_index']):04d}__{row['issue_label']}__{snapshot.dataset_label}"
            f"__{snapshot.source_name}__example_{int(snapshot.example_id):03d}"
            f"__timestep_{int(snapshot.timestep):02d}"
        )
        _plot_snapshot(
            out_path=stem,
            dataset_label=str(snapshot.dataset_label),
            review_index=int(row["review_index"]),
            issue_label=str(row["issue_label"]),
            example_id=int(snapshot.example_id),
            timestep=int(snapshot.timestep),
            source_frame=frame,
            fluid_mask=fluid_mask,
            channels=channels,
            source_name=str(snapshot.source_name),
            gradient_field=str(row["dominant_gradient_field"]),
            oscillation_field=str(row["dominant_oscillation_field"]),
            row=row,
        )


@hydra.main(version_base=None, config_path="conf", config_name="assess_problematic_examples_generic")
def main(cfg: DictConfig) -> None:
    cfg = hack_conf(cfg)
    out_dir = env.out()
    with (out_dir / "conf_assess_problematic_examples.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    input_source = instantiate(cfg.problematic_input_source)
    audit_fields = tuple(str(x) for x in cfg.analysis.audit_fields)
    channels = {
        "p": int(cfg.analysis.p_channel),
        "ux": int(cfg.analysis.ux_channel),
        "uy": int(cfg.analysis.uy_channel),
    }
    fluid_value = int(cfg.analysis.fluid_value)
    min_valid_points = int(cfg.analysis.min_valid_points)
    score_quantile = float(cfg.analysis.score_quantile)
    normalize_gradient_by_example_speed = bool(cfg.analysis.normalize_gradient_by_example_speed)
    speed_reference_quantile = float(cfg.analysis.speed_reference_quantile)
    speed_reference_eps = float(cfg.analysis.speed_reference_eps)

    with input_source:
        metrics_df = _collect_metrics(
            input_source,
            audit_fields=audit_fields,
            channels=channels,
            fluid_value=fluid_value,
            min_valid_points=min_valid_points,
            score_quantile=score_quantile,
            normalize_gradient_by_example_speed=normalize_gradient_by_example_speed,
            speed_reference_quantile=speed_reference_quantile,
            speed_reference_eps=speed_reference_eps,
        )
        metrics_df.to_csv(out_dir / str(cfg.outputs.snapshot_metrics_csv), index=False)

        selected_df = _rank_selected(
            metrics_df,
            grouping=str(cfg.selection.grouping),
            top_k_gradient=int(cfg.selection.top_k_gradient),
            top_k_oscillation=int(cfg.selection.top_k_oscillation),
            min_gradient_score=float(cfg.selection.min_gradient_score),
            min_oscillation_score=float(cfg.selection.min_oscillation_score),
            snapshots_per_example=int(cfg.selection.snapshots_per_example),
        )
        selected_df.to_csv(out_dir / str(cfg.outputs.selected_manifest_csv), index=False)

        plot_dir = out_dir / str(cfg.outputs.plot_subdir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_selected_snapshots(
            input_source,
            selected_df,
            plot_dir=plot_dir,
            channels=channels,
            fluid_value=fluid_value,
        )

    print(f"Snapshot metrics       -> {out_dir / str(cfg.outputs.snapshot_metrics_csv)}")
    print(f"Selected manifest      -> {out_dir / str(cfg.outputs.selected_manifest_csv)}")
    print(f"Suspicious snapshots   -> {out_dir / str(cfg.outputs.plot_subdir)}")


if __name__ == "__main__":
    main()
