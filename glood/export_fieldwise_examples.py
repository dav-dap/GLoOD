from __future__ import annotations

import math
import tarfile
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env


def _build_dataset_index(cfg: DictConfig) -> dict[str, dict[str, str | int]]:
    index: dict[str, dict[str, str | int]] = {}
    for dataset_cfg in cfg.datasets:
        label = str(dataset_cfg.label)
        index[label] = {
            "input_dir": str(dataset_cfg.input_dir),
            "example_id_offset": int(getattr(dataset_cfg, "example_id_offset", 0) or 0),
        }
    return index


def _member_name(
    *,
    dataset_label: str,
    tar_example_id: int,
    epoch: int,
    timestep: int,
    kind: str,
) -> str:
    prefix = f"{dataset_label}/example_{tar_example_id}/epoch_{epoch}/timesep_{timestep}"
    if kind == "fieldwise_panel":
        return f"{prefix}/timestep_{timestep}.png"
    if kind == "u_true":
        return f"{prefix}/partials/{timestep}_u_true.png"
    if kind == "u_pred":
        return f"{prefix}/partials/{timestep}_u_pred.png"
    if kind == "u_mse":
        return f"{prefix}/partials/{timestep}_u_mse.png"
    if kind == "u_cos_dist":
        return f"{prefix}/partials/{timestep}_u_cos_dist.png"
    if kind == "p_true":
        return f"{prefix}/partials/{timestep}_p_true.png"
    if kind == "p_pred":
        return f"{prefix}/partials/{timestep}_p_pred.png"
    if kind == "p_mse":
        return f"{prefix}/partials/{timestep}_p_mse.png"
    raise ValueError(f"Unsupported artifact kind: {kind}")


def _safe_slug(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def _target_name(
    *,
    dataset_label: str,
    example_id: int,
    timestep: int,
    kind: str,
    category: str | None,
) -> str:
    category_part = f"{_safe_slug(category)}_" if category else ""
    return f"{dataset_label}_{category_part}example_{example_id:03d}_timestep_{timestep:02d}_{kind}.png"


def _candidate_tar_example_ids(example_id: int, example_id_offset: int) -> list[int]:
    candidates = [example_id]
    if example_id_offset != 0:
        shifted = example_id - example_id_offset
        if shifted not in candidates and shifted > 0:
            candidates.append(shifted)
    return candidates


def _load_manifest_inputs(cfg: DictConfig) -> pd.DataFrame:
    manifest_path = Path(str(cfg.inputs.manifest_csv))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        raise ValueError(f"Manifest CSV is empty: {manifest_path}")
    if "category" not in manifest_df.columns:
        manifest_df["category"] = ""
    return manifest_df


def _prepare_diagnostics_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Diagnostics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Diagnostics CSV is empty: {csv_path}")
    df["abs_best_shift_dx"] = df["best_shift_dx"].abs()
    df["abs_best_shift_dy"] = df["best_shift_dy"].abs()
    df["best_shift_manhattan"] = df["abs_best_shift_dx"] + df["abs_best_shift_dy"]
    df["abs_best_shift_scale_dx"] = df["best_shift_scale_dx"].abs()
    df["abs_best_shift_scale_dy"] = df["best_shift_scale_dy"].abs()
    df["best_shift_scale_manhattan"] = df["abs_best_shift_scale_dx"] + df["abs_best_shift_scale_dy"]
    return df


def _apply_filter(df: pd.DataFrame, *, column: str, op: str, value: float | int | str) -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(f"Selection filter references missing column: {column}")
    if op == "eq":
        return df[df[column] == value]
    if op == "ne":
        return df[df[column] != value]
    if op == "gt":
        return df[df[column] > value]
    if op == "ge":
        return df[df[column] >= value]
    if op == "lt":
        return df[df[column] < value]
    if op == "le":
        return df[df[column] <= value]
    raise ValueError(f"Unsupported filter op: {op}")


def _select_paired_examples(
    filtered: pd.DataFrame,
    *,
    label: str,
    top_k_pairs: int,
    sort_by: str,
    ascending: bool,
    pair_group_columns: list[str],
    required_models: list[str],
    model_column: str,
) -> pd.DataFrame:
    if model_column not in filtered.columns:
        raise KeyError(f"Selection category {label!r} requires missing model column: {model_column}")
    for column in pair_group_columns:
        if column not in filtered.columns:
            raise KeyError(f"Selection category {label!r} requires missing pair-group column: {column}")

    pieces: list[pd.DataFrame] = []
    scored_groups: list[tuple[float, tuple[object, ...], pd.DataFrame]] = []
    for key, sub_df in filtered.groupby(pair_group_columns, dropna=False, sort=False):
        present_models = {str(x) for x in sub_df[model_column].unique()}
        if not set(required_models).issubset(present_models):
            continue
        pair_df = sub_df[sub_df[model_column].isin(required_models)].copy()
        score = float(pair_df[sort_by].mean())
        scored_groups.append((score, key if isinstance(key, tuple) else (key,), pair_df))

    if not scored_groups:
        return pd.DataFrame(columns=filtered.columns)

    scored_groups.sort(key=lambda item: (item[0], item[1]), reverse=not ascending)
    for _, _, pair_df in scored_groups[:top_k_pairs]:
        ordered = pair_df.sort_values([model_column, "dataset_label", "example_id"], kind="mergesort")
        pieces.append(ordered)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=filtered.columns)


def _select_from_diagnostics(cfg: DictConfig) -> pd.DataFrame:
    diagnostics_path = Path(str(cfg.inputs.diagnostics_csv))
    df = _prepare_diagnostics_df(diagnostics_path)

    field_name = str(cfg.selection.field_name)
    timestep = int(cfg.selection.timestep)
    epoch = int(cfg.export.epoch)
    base_df = df[
        (df["field_name"] == field_name)
        & (df["timestep"] == timestep)
        & (df["epoch"] == epoch)
    ].copy()
    if base_df.empty:
        raise ValueError(
            f"No diagnostics rows left after filtering field_name={field_name}, timestep={timestep}, epoch={epoch}"
        )

    group_by_columns = [str(x) for x in getattr(cfg.selection, "group_by_columns", [])]
    categories: list[pd.DataFrame] = []
    for category_cfg in cfg.selection.categories:
        label = str(category_cfg.label)
        top_k = int(getattr(category_cfg, "top_k", 1))
        sort_by = str(category_cfg.sort_by)
        ascending = bool(getattr(category_cfg, "ascending", False))
        pair_models = bool(getattr(category_cfg, "pair_models", False))
        filtered = base_df.copy()
        for filter_cfg in getattr(category_cfg, "filters", []):
            filtered = _apply_filter(
                filtered,
                column=str(filter_cfg.column),
                op=str(filter_cfg.op),
                value=filter_cfg.value,
            )
        if filtered.empty:
            continue
        if sort_by not in filtered.columns:
            raise KeyError(f"Selection category {label!r} uses missing sort column: {sort_by}")
        filtered = filtered.sort_values(
            [sort_by, "dataset_label", "example_id"],
            ascending=[ascending, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        if pair_models:
            pair_group_columns = [str(x) for x in getattr(category_cfg, "pair_group_columns", ["geometry", "example_id"])]
            required_models = [str(x) for x in getattr(category_cfg, "required_models", ["swin", "unet"])]
            model_column = str(getattr(category_cfg, "model_column", "model"))
            top_k_pairs = int(getattr(category_cfg, "top_k_pairs", top_k))
            selected = _select_paired_examples(
                filtered,
                label=label,
                top_k_pairs=top_k_pairs,
                sort_by=sort_by,
                ascending=ascending,
                pair_group_columns=pair_group_columns,
                required_models=required_models,
                model_column=model_column,
            )
        elif group_by_columns:
            pieces: list[pd.DataFrame] = []
            for _, sub_df in filtered.groupby(group_by_columns, dropna=False, sort=False):
                pieces.append(sub_df.head(top_k))
            selected = pd.concat(pieces, ignore_index=True)
        else:
            selected = filtered.head(top_k).copy()

        if selected.empty:
            continue
        selected["category"] = label
        categories.append(selected)

    if not categories:
        raise ValueError("No examples matched the configured diagnostic categories")

    selected_df = pd.concat(categories, ignore_index=True)
    keep_cols = [
        "category",
        "model",
        "geometry",
        "dataset_label",
        "example_id",
        "epoch",
        "timestep",
        "field_name",
        "raw_rel_l2",
        "raw_pearson_corr",
        "scale_explained_fraction",
        "shift_explained_fraction",
        "shift_scale_explained_fraction",
        "structural_residual_ratio",
        "best_shift_dx",
        "best_shift_dy",
        "best_shift_scale_dx",
        "best_shift_scale_dy",
        "best_shift_scale_manhattan",
    ]
    keep_cols = [c for c in keep_cols if c in selected_df.columns]
    return selected_df[keep_cols].reset_index(drop=True)


def _select_examples(cfg: DictConfig) -> pd.DataFrame:
    mode = str(getattr(cfg.inputs, "selection_source", "manifest_csv")).strip().lower()
    if mode == "manifest_csv":
        return _load_manifest_inputs(cfg)
    if mode == "diagnostics_csv":
        return _select_from_diagnostics(cfg)
    raise ValueError(f"Unsupported inputs.selection_source: {mode}")


@hydra.main(version_base=None, config_path="conf", config_name="export_fieldwise_examples_generic")
def main(cfg: DictConfig) -> None:
    out_dir = env.out()
    with (out_dir / "conf_export_fieldwise_examples.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    selected_df = _select_examples(cfg)
    dataset_index = _build_dataset_index(cfg)
    kinds = tuple(str(x) for x in cfg.export.artifacts)
    timestep = int(cfg.export.timestep)
    epoch = int(cfg.export.epoch)
    output_dir = out_dir / str(cfg.export.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_df.to_csv(out_dir / "selected_examples_manifest.csv", index=False)

    for dataset_label, sub_df in selected_df.groupby("dataset_label", dropna=False):
        dataset_label = str(dataset_label)
        dataset_meta = dataset_index.get(dataset_label)
        if dataset_meta is None:
            raise KeyError(f"Dataset label {dataset_label!r} is not present in cfg.datasets")

        tar_path = Path(str(dataset_meta["input_dir"]))
        example_id_offset = int(dataset_meta["example_id_offset"])
        if not tar_path.is_file():
            raise FileNotFoundError(f"Input tarball not found: {tar_path}")

        with tarfile.open(tar_path) as tf:
            for row in sub_df.itertuples(index=False):
                example_id = int(row.example_id)
                category = None
                if hasattr(row, "category"):
                    raw_category = getattr(row, "category")
                    if raw_category is not None and not (isinstance(raw_category, float) and math.isnan(raw_category)):
                        category = str(raw_category)
                for kind in kinds:
                    src = None
                    attempted_members: list[str] = []
                    for tar_example_id in _candidate_tar_example_ids(example_id, example_id_offset):
                        member = _member_name(
                            dataset_label=dataset_label,
                            tar_example_id=tar_example_id,
                            epoch=epoch,
                            timestep=timestep,
                            kind=kind,
                        )
                        attempted_members.append(member)
                        try:
                            src = tf.extractfile(member)
                        except KeyError:
                            src = None
                        if src is not None:
                            break
                    if src is None:
                        attempted = ", ".join(attempted_members)
                        raise FileNotFoundError(
                            f"Member not found in {tar_path} for example_id={example_id}; attempted: {attempted}"
                        )
                    target_path = output_dir / _target_name(
                        dataset_label=dataset_label,
                        example_id=example_id,
                        timestep=timestep,
                        kind=kind,
                        category=category,
                    )
                    target_path.write_bytes(src.read())


if __name__ == "__main__":
    main()
