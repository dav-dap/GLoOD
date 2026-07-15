from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env
from glood.assess_drag_force import (
    FORCE_COMPONENTS,
    TERCILE_LABELS,
)
from glood.src.assessment.force_metrics import compute_series_force_components
from glood.src.assessment.input_materialization import materialized_inference_input
from glood.src.assessment.mask_metrics import ensure_series_axis

PLOT_COMPONENTS = ("Fx_p", "Fy_p", "Fx_v_raw", "Fy_v_raw")


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        col
        for col in df.columns
        if col.startswith("pred_")
        or col.startswith("true_")
        or col.startswith("err_")
        or col.startswith("rel_err_")
        or col.startswith("abs_rel_err_")
    ]
    summary = df[cols].agg(["mean", "std"]).transpose().reset_index()
    summary.columns = ["metric", "mean", "std"]
    return summary


def _add_normalized_errors(df: pd.DataFrame, *, error_epsilon: float) -> pd.DataFrame:
    out = df.copy()
    for comp in FORCE_COMPONENTS:
        denom = np.abs(out[f"true_{comp}"].to_numpy(dtype=float)) + float(error_epsilon)
        err = out[f"err_{comp}"].to_numpy(dtype=float)
        rel = err / denom
        out[f"rel_err_{comp}"] = rel
        out[f"abs_rel_err_{comp}"] = np.abs(rel)
    return out


def _build_global(df: pd.DataFrame, *, viscous_prefactor: float, error_epsilon: float) -> pd.DataFrame:
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


def _compute_prefix_exclusion_stats(df: pd.DataFrame, *, components: tuple[str, ...]) -> pd.DataFrame:
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
                    "abs_rel_err_median": float(np.median(abs_rel)),
                    "abs_rel_err_p90": float(np.quantile(abs_rel, 0.90)),
                    "abs_rel_err_p95": float(np.quantile(abs_rel, 0.95)),
                    "abs_rel_err_p99": float(np.quantile(abs_rel, 0.99)),
                }
            )
    return pd.DataFrame(rows).sort_values(["component", "exclude_first_k"]).reset_index(drop=True)


def _compute_magnitude_gated_stats(
    df: pd.DataFrame,
    *,
    components: tuple[str, ...],
    true_force_percentiles: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for comp in components:
        true_abs = np.abs(df[f"true_{comp}"].to_numpy(dtype=float))
        rel = df[f"rel_err_{comp}"].to_numpy(dtype=float)
        abs_rel = df[f"abs_rel_err_{comp}"].to_numpy(dtype=float)
        total_n = int(len(true_abs))
        for percentile in true_force_percentiles:
            p_clamped = float(np.clip(float(percentile), 0.0, 100.0))
            thr = float(np.quantile(true_abs, p_clamped / 100.0))
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
    columns = [
        "component",
        "target_percentile",
        "true_force_threshold",
        "tau_percentile_empirical",
        "n_rows",
        "retained_fraction",
        "rel_err_mean",
        "abs_rel_err_mean",
        "abs_rel_err_median",
        "abs_rel_err_p75",
        "abs_rel_err_p90",
        "abs_rel_err_p95",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["component", "target_percentile"]).reset_index(drop=True)


def _compute_example_last_timestep_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["example_id", "epoch", "last_timestep", "score_pressure", "score_viscous_raw"]
        )
    max_timestep = df.groupby(["example_id", "epoch"], as_index=False)["timestep"].max()
    max_timestep = max_timestep.rename(columns={"timestep": "last_timestep"})
    merged = df.merge(
        max_timestep,
        left_on=["example_id", "epoch", "timestep"],
        right_on=["example_id", "epoch", "last_timestep"],
        how="inner",
    )
    out = merged[["example_id", "epoch", "last_timestep"]].copy()
    out["score_pressure"] = (
        merged["true_Fx_p"].to_numpy(dtype=float) ** 2 + merged["true_Fy_p"].to_numpy(dtype=float) ** 2
    )
    out["score_viscous_raw"] = (
        merged["true_Fx_v_raw"].to_numpy(dtype=float) ** 2 + merged["true_Fy_v_raw"].to_numpy(dtype=float) ** 2
    )
    return out


def _assign_terciles(scores: pd.Series) -> pd.Series:
    ranked = scores.rank(method="first")
    bins = pd.qcut(ranked, q=3, labels=list(TERCILE_LABELS))
    return bins.astype(str)


def _write_analysis_bundle(df: pd.DataFrame, *, output_dir: Path, cfg: DictConfig) -> None:
    summary_df = _build_summary(df)
    global_df = _build_global(
        df,
        viscous_prefactor=float(cfg.assessment.viscous_prefactor),
        error_epsilon=float(cfg.assessment.error_epsilon),
    )
    prefix_df = _compute_prefix_exclusion_stats(df, components=PLOT_COMPONENTS)
    magnitude_df = _compute_magnitude_gated_stats(
        df,
        components=PLOT_COMPONENTS,
        true_force_percentiles=tuple(float(x) for x in cfg.analysis.true_force_percentiles),
    )

    summary_df.to_csv(output_dir / cfg.outputs.summary_csv, index=False)
    global_df.to_csv(output_dir / cfg.outputs.global_csv, index=False)
    prefix_df.to_csv(output_dir / cfg.outputs.prefix_exclusion_csv, index=False)
    magnitude_df.to_csv(output_dir / cfg.outputs.magnitude_gated_csv, index=False)


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


def _resolve_input_dir(input_path: Path, *, label: str) -> Path:
    return input_path / "infer"


def _compute_per_timestep(input_dir: Path, *, cfg: DictConfig, example_id_offset: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for example_id, epoch, data_dir in _iter_example_dirs(input_dir):
        pred = ensure_series_axis(np.load(data_dir / "pred.npy", allow_pickle=False, mmap_mode="r"), name="pred")
        true = ensure_series_axis(np.load(data_dir / "true.npy", allow_pickle=False, mmap_mode="r"), name="true")
        matnum = np.load(data_dir / "matnum.npy", allow_pickle=False, mmap_mode="r")

        pred_series = compute_series_force_components(
            pred,
            matnum,
            p_channel=int(cfg.assessment.p_channel),
            ux_channel=int(cfg.assessment.ux_channel),
            uy_channel=int(cfg.assessment.uy_channel),
            fluid_value=int(cfg.assessment.fluid_value),
            wall_value=int(cfg.assessment.wall_value),
            crop_margin=int(cfg.assessment.crop_margin),
            dx=float(cfg.assessment.dx),
            dy=float(cfg.assessment.dy),
            viscous_prefactor=float(cfg.assessment.viscous_prefactor),
        )
        true_series = compute_series_force_components(
            true,
            matnum,
            p_channel=int(cfg.assessment.p_channel),
            ux_channel=int(cfg.assessment.ux_channel),
            uy_channel=int(cfg.assessment.uy_channel),
            fluid_value=int(cfg.assessment.fluid_value),
            wall_value=int(cfg.assessment.wall_value),
            crop_margin=int(cfg.assessment.crop_margin),
            dx=float(cfg.assessment.dx),
            dy=float(cfg.assessment.dy),
            viscous_prefactor=float(cfg.assessment.viscous_prefactor),
        )

        normalized_example_id = int(example_id) + int(example_id_offset)
        for timestep, (pred_comp, true_comp) in enumerate(zip(pred_series, true_series)):
            row: dict[str, float | int] = {
                "example_id": normalized_example_id,
                "epoch": int(epoch),
                "timestep": int(timestep),
                "viscous_prefactor": float(cfg.assessment.viscous_prefactor),
                "pred_interface_faces": int(pred_comp["interface_faces"]),
                "true_interface_faces": int(true_comp["interface_faces"]),
            }
            for comp in FORCE_COMPONENTS:
                pred_value = float(pred_comp[comp])
                true_value = float(true_comp[comp])
                row[f"pred_{comp}"] = pred_value
                row[f"true_{comp}"] = true_value
                row[f"err_{comp}"] = pred_value - true_value
            rows.append(row)

    if not rows:
        raise ValueError(f"No valid data rows found in {input_dir}")
    return pd.DataFrame(rows)


def _write_dataset_outputs(label: str, input_dir: Path, *, cfg: DictConfig, example_id_offset: int, root_out: Path) -> None:
    out_dir = root_out / label
    out_dir.mkdir(parents=True, exist_ok=True)

    with materialized_inference_input(input_dir, label=label, cfg=cfg) as resolved_input_dir:
        per_timestep_df = _compute_per_timestep(
            resolved_input_dir,
            cfg=cfg,
            example_id_offset=example_id_offset,
        ).sort_values(["example_id", "epoch", "timestep"]).reset_index(drop=True)
    analysis_df = _add_normalized_errors(per_timestep_df, error_epsilon=float(cfg.assessment.error_epsilon))
    per_timestep_df.to_csv(out_dir / cfg.outputs.per_timestep_csv, index=False)
    _write_analysis_bundle(analysis_df, output_dir=out_dir, cfg=cfg)

    score_df = _compute_example_last_timestep_scores(per_timestep_df)
    score_df["tercile_pressure"] = _assign_terciles(score_df["score_pressure"])
    score_df["tercile_viscous_raw"] = _assign_terciles(score_df["score_viscous_raw"])
    score_df.to_csv(out_dir / cfg.outputs.example_terciles_csv, index=False)

    for strat_col, strat_name in (("tercile_pressure", "pressure"), ("tercile_viscous_raw", "viscous_raw")):
        for tercile in TERCILE_LABELS:
            sub_keys = score_df[score_df[strat_col] == tercile][["example_id", "epoch"]].drop_duplicates()
            if sub_keys.empty:
                continue
            sub_raw_df = per_timestep_df.merge(sub_keys, on=["example_id", "epoch"], how="inner")
            sub_df = analysis_df.merge(sub_keys, on=["example_id", "epoch"], how="inner")
            sub_dir = out_dir / str(cfg.outputs.stratified_subdir) / strat_name / tercile
            sub_dir.mkdir(parents=True, exist_ok=True)
            sub_raw_df.to_csv(sub_dir / cfg.outputs.per_timestep_csv, index=False)
            _write_analysis_bundle(sub_df, output_dir=sub_dir, cfg=cfg)


@hydra.main(version_base=None, config_path="conf", config_name="assess_drag_force_remote_generic")
def main(cfg: DictConfig) -> None:
    out_dir = env.out()
    with (out_dir / "conf_assess_drag_force_remote.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    for dataset_cfg in cfg.datasets:
        label = str(dataset_cfg.label)
        geometry = str(getattr(dataset_cfg, "geometry", "")).strip().lower()
        if geometry == "cavity":
            print(f"SKIP {label}: drag-force assessment is not defined for geometry=cavity")
            continue
        input_dir = Path(str(dataset_cfg.input_dir))
        example_id_offset = int(getattr(dataset_cfg, "example_id_offset", 0))
        print(f"RERUN {label} from {input_dir}")
        _write_dataset_outputs(
            label,
            input_dir,
            cfg=cfg,
            example_id_offset=example_id_offset,
            root_out=out_dir,
        )


if __name__ == "__main__":
    main()
