from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env
from glood.src.assessment.mask_metrics import (
    METRIC_COLUMNS,
    binary_metrics,
    binary_metrics_from_counts,
    ensure_series_axis,
    matnum_activity_mask,
    velocity_activity_mask,
)
from glood.src.assessment.provider_utils import instantiate_providers


def _build_pair_rows(
    *,
    example_id: int,
    epoch: int,
    pair: str,
    epsilon: float,
    lhs_mask,
    rhs_mask,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    if lhs_mask.shape != rhs_mask.shape:
        raise ValueError(f"Mask shape mismatch for {pair}: {lhs_mask.shape} vs {rhs_mask.shape}.")

    for timestep in range(lhs_mask.shape[0]):
        metrics = binary_metrics(lhs_mask[timestep], rhs_mask[timestep])
        rows.append(
            {
                "example_id": example_id,
                "epoch": epoch,
                "timestep": timestep,
                "pair": pair,
                "epsilon": float(epsilon),
                "lhs_true_fraction": float(lhs_mask[timestep].mean()),
                "rhs_true_fraction": float(rhs_mask[timestep].mean()),
                **metrics,
            }
        )
    return rows


def _summarize_by_pair(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["lhs_true_fraction", "rhs_true_fraction", *METRIC_COLUMNS]
    summary = df.groupby("pair", dropna=False)[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "pair" if col == ("pair", "") else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    summary.insert(1, "epsilon", float(df["epsilon"].iloc[0]))
    return summary


def _global_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    epsilon = float(df["epsilon"].iloc[0])
    for pair, group in df.groupby("pair", dropna=False):
        tp = int(group["tp"].sum())
        tn = int(group["tn"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        metrics = binary_metrics_from_counts(tp, tn, fp, fn)
        rows.append(
            {
                "pair": pair,
                "epsilon": epsilon,
                "n_rows": int(len(group)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("pair").reset_index(drop=True)


@hydra.main(version_base=None, config_path="conf", config_name="assess_boundaries_generic")
def main(cfg: DictConfig) -> None:
    with (env.out() / "conf_assess_boundaries.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    providers = instantiate_providers(cfg.plotting_providers)

    eps = float(cfg.assessment.epsilon)
    ux_channel = int(cfg.assessment.ux_channel)
    uy_channel = int(cfg.assessment.uy_channel)
    matnum_value = int(cfg.assessment.matnum_value)
    require_matnum = bool(cfg.assessment.require_matnum)

    rows: list[dict[str, float | int | str]] = []
    assessed_examples = 0

    for provider in providers:
        for plottable_input in provider:
            if not hasattr(plottable_input, "pred") or not hasattr(plottable_input, "true"):
                raise TypeError(
                    "Plotting provider must yield objects with 'pred' and 'true' attributes."
                )
            pred = ensure_series_axis(plottable_input.pred, name="pred")
            true = ensure_series_axis(plottable_input.true, name="true")
            if pred.shape != true.shape:
                raise ValueError(
                    f"Pred/true shape mismatch for example {plottable_input.id}: {pred.shape} vs {true.shape}."
                )

            pred_mask = velocity_activity_mask(
                pred,
                eps=eps,
                ux_channel=ux_channel,
                uy_channel=uy_channel,
            )
            true_mask = velocity_activity_mask(
                true,
                eps=eps,
                ux_channel=ux_channel,
                uy_channel=uy_channel,
            )

            rows.extend(
                _build_pair_rows(
                    example_id=plottable_input.id,
                    epoch=plottable_input.epoch,
                    pair="pred_vs_true",
                    epsilon=eps,
                    lhs_mask=pred_mask,
                    rhs_mask=true_mask,
                )
            )

            if plottable_input.matnum is None:
                if require_matnum:
                    raise ValueError(
                        f"matnum is required but missing for example {plottable_input.id}, epoch {plottable_input.epoch}."
                    )
            else:
                matnum_mask = matnum_activity_mask(
                    plottable_input.matnum,
                    series_len=pred.shape[0],
                    matnum_value=matnum_value,
                )
                rows.extend(
                    _build_pair_rows(
                        example_id=plottable_input.id,
                        epoch=plottable_input.epoch,
                        pair="true_vs_matnum",
                        epsilon=eps,
                        lhs_mask=true_mask,
                        rhs_mask=matnum_mask,
                    )
                )
                rows.extend(
                    _build_pair_rows(
                        example_id=plottable_input.id,
                        epoch=plottable_input.epoch,
                        pair="pred_vs_matnum",
                        epsilon=eps,
                        lhs_mask=pred_mask,
                        rhs_mask=matnum_mask,
                    )
                )

            assessed_examples += 1

    if not rows:
        raise ValueError("No examples were assessed. Check provider/input configuration.")

    output_dir = Path(env.out())
    output_dir.mkdir(parents=True, exist_ok=True)

    per_timestep_df = pd.DataFrame(rows)
    by_pair_df = _summarize_by_pair(per_timestep_df)
    global_df = _global_summary(per_timestep_df)

    per_timestep_path = output_dir / str(cfg.outputs.per_timestep_csv)
    by_pair_path = output_dir / str(cfg.outputs.by_pair_csv)
    global_path = output_dir / str(cfg.outputs.global_csv)

    per_timestep_df.to_csv(per_timestep_path, index=False)
    by_pair_df.to_csv(by_pair_path, index=False)
    global_df.to_csv(global_path, index=False)

    print(f"Assessed examples: {assessed_examples}")
    print(f"Per-timestep metrics -> {per_timestep_path.name}")
    print(f"Pair summary        -> {by_pair_path.name}")
    print(f"Global summary      -> {global_path.name}")


if __name__ == "__main__":
    main()
